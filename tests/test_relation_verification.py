import json

import pytest

from knowledge_forge.config import Config
from knowledge_forge.relation_verification import (
    PUBLISHABLE_RELATION_STATUSES,
    is_publishable_relation,
    relation_candidate_id,
)
from knowledge_forge.schema_evolution import BASE_SCHEMA, SchemaEvolutionService


SOURCE = "Alpha produces Beta. Beta uses Gamma."


def extraction(relations=None):
    alpha_start = SOURCE.index("Alpha")
    beta_start = SOURCE.index("Beta")
    gamma_start = SOURCE.index("Gamma")
    return {
        "entities": [
            {"id": "e1", "name": "Alpha", "type": "Organization", "description": "", "aliases": [],
             "evidence": "Alpha", "mention_start": alpha_start,
             "mention_end": alpha_start + len("Alpha"), "confidence": 0.9},
            {"id": "e2", "name": "Beta", "type": "Product", "description": "", "aliases": [],
             "evidence": "Beta", "mention_start": beta_start,
             "mention_end": beta_start + len("Beta"), "confidence": 0.9},
            {"id": "e3", "name": "Gamma", "type": "Technology", "description": "", "aliases": [],
             "evidence": "Gamma", "mention_start": gamma_start,
             "mention_end": gamma_start + len("Gamma"), "confidence": 0.9},
        ],
        "relations": relations if relations is not None else [
            {"source": "e1", "target": "e2", "type": "produces", "description": "",
             "evidence": "Alpha produces Beta.", "confidence": 0.9},
        ],
    }


class ScriptedVerifier:
    def __init__(self, extracted, callbacks):
        self.extracted = extracted
        self.callbacks = list(callbacks)
        self.calls = []
        self.verification_payloads = []

    def json_chat(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return self.extracted
        marker = "Candidate relations:\n"
        payload = json.loads(kwargs["prompt"].split(marker, 1)[1].split("\n\nSOURCE TEXT:", 1)[0])
        self.verification_payloads.append(payload)
        callback = self.callbacks.pop(0)
        if isinstance(callback, BaseException):
            raise callback
        return callback(payload)


def decision(candidate, verdict="supported", *, relation_type=None, orientation="as_is", evidence=None, reason="verified"):
    return {
        "candidate_id": candidate["candidate_id"],
        "verdict": verdict,
        "corrected_type": relation_type or candidate["proposed_type"],
        "orientation": orientation,
        "evidence": candidate["proposed_evidence"] if evidence is None else evidence,
        "reason": reason,
    }


def run_extract(monkeypatch, ollama, *, source=SOURCE, extracted_chunk=None, verification=True):
    monkeypatch.setattr(Config, "extraction_gleaning_enabled", False)
    monkeypatch.setattr(Config, "extraction_verification_enabled", verification)
    chunk = extracted_chunk or {
        "id": 9,
        "chunk_key": "chunk-1",
        "content": source,
        "metadata": {"section_key": "s1", "section_title": "Section"},
    }
    return SchemaEvolutionService(None, ollama).extract_chunk(
        {"id": 4, "project_key": "project-4", "llm_model": "test", "model_max_tokens": 32768},
        {"version": 2, "schema_json": BASE_SCHEMA},
        chunk,
    )


def test_stable_candidate_id_is_deterministic_and_project_scoped():
    values = dict(
        chunk_key="chunk-1", source_mention="e1", target_mention="e2",
        proposed_type="uses", ordinal=0, evidence_start=2, evidence_end=10,
        evidence="Alpha uses Beta",
    )
    assert relation_candidate_id(project_scope="p1", **values) == relation_candidate_id(project_scope="p1", **values)
    assert relation_candidate_id(project_scope="p1", **values) != relation_candidate_id(project_scope="p2", **values)
    assert relation_candidate_id(project_scope="p1", **values) != relation_candidate_id(
        project_scope="p1", **{**values, "chunk_key": "chunk-2"},
    )


def test_empty_verifier_response_retries_then_marks_missing(monkeypatch):
    ollama = ScriptedVerifier(extraction(), [lambda _: {"decisions": []}, lambda _: {"decisions": []}])
    result = run_extract(monkeypatch, ollama)
    assert result["relations"][0]["verification_status"] == "verification_missing"
    assert not is_publishable_relation(result["relations"][0]["verification_status"])
    assert result["_diagnostics"]["retry_count"] == 1
    assert result["_diagnostics"]["missing_decisions"] == 1
    assert len(result["_verification_events"]) == 2
    assert all(call["context_budget_tokens"] == 32768 for call in ollama.calls)
    assert all("max_output_tokens" in call for call in ollama.calls)
    assert all("max_tokens" not in call for call in ollama.calls)


def test_only_missing_candidate_is_retried_and_remains_quarantined(monkeypatch):
    extracted = extraction([
        {"source": "e1", "target": "e2", "type": "produces", "description": "",
         "evidence": "Alpha produces Beta.", "confidence": 0.9},
        {"source": "e2", "target": "e3", "type": "uses", "description": "",
         "evidence": "Beta uses Gamma.", "confidence": 0.9},
    ])
    ollama = ScriptedVerifier(extracted, [
        lambda values: {"decisions": [decision(values[0])]},
        lambda _: {"decisions": []},
    ])
    result = run_extract(monkeypatch, ollama)
    assert [item["verification_status"] for item in result["relations"]] == [
        "llm_verified", "verification_missing",
    ]
    assert len(ollama.verification_payloads[0]) == 2
    assert len(ollama.verification_payloads[1]) == 1
    assert ollama.verification_payloads[1][0]["candidate_id"] == result["relations"][1]["candidate_id"]


@pytest.mark.parametrize("mode", ["duplicate", "unknown"])
def test_duplicate_or_unknown_candidate_id_fails_closed(monkeypatch, mode):
    def invalid(values):
        item = decision(values[0])
        if mode == "unknown":
            item["candidate_id"] = "not-requested"
            return {"decisions": [item]}
        return {"decisions": [item, dict(item)]}

    result = run_extract(monkeypatch, ScriptedVerifier(extraction(), [invalid]))
    assert result["relations"][0]["verification_status"] == "verification_failed"
    assert result["_diagnostics"]["published"] == 0
    assert result["_verification_events"][0]["error_category"] == f"{mode}_candidate_id"


@pytest.mark.parametrize("failure", [TimeoutError("timeout"), RuntimeError("invalid JSON")])
def test_verifier_exception_or_invalid_json_fails_closed(monkeypatch, failure):
    result = run_extract(monkeypatch, ScriptedVerifier(extraction(), [failure]))
    assert result["relations"][0]["verification_status"] == "verification_failed"
    assert result["_diagnostics"]["verifier_failures"] == 1
    assert result["_verification_events"][0]["error_category"] == "verifier_exception"


@pytest.mark.parametrize(
    "response",
    [
        {"decisions": [{"candidate_id": "placeholder"}]},
        {"decisions": [], "unexpected": True},
    ],
)
def test_malformed_structured_response_fails_closed(monkeypatch, response):
    def malformed(values):
        value = json.loads(json.dumps(response))
        decisions = value.get("decisions") or []
        if decisions:
            decisions[0]["candidate_id"] = values[0]["candidate_id"]
        return value

    result = run_extract(monkeypatch, ScriptedVerifier(extraction(), [malformed]))
    assert result["relations"][0]["verification_status"] == "verification_failed"
    assert result["_diagnostics"]["published"] == 0
    assert result["_verification_events"][0]["error_category"] == "invalid_decisions"


def test_non_object_verifier_response_fails_closed(monkeypatch):
    result = run_extract(monkeypatch, ScriptedVerifier(extraction(), [lambda _: []]))
    assert result["relations"][0]["verification_status"] == "verification_failed"
    assert result["_diagnostics"]["published"] == 0
    assert result["_verification_events"][0]["error_category"] == "invalid_decisions"


def test_supported_relation_with_exact_evidence_is_publishable(monkeypatch):
    result = run_extract(monkeypatch, ScriptedVerifier(
        extraction(), [lambda values: {"decisions": [decision(values[0])]}],
    ))
    relation = result["relations"][0]
    assert relation["verification_status"] == "llm_verified"
    assert relation["evidence_text"] == "Alpha produces Beta."
    assert relation["evidence_start"] == 0
    assert is_publishable_relation(relation["verification_status"], relation["metadata"])


def test_supported_relation_may_omit_corrected_type(monkeypatch):
    def without_correction(values):
        item = decision(values[0])
        item.pop("corrected_type")
        return {"decisions": [item]}

    result = run_extract(monkeypatch, ScriptedVerifier(extraction(), [without_correction]))
    assert result["relations"][0]["verification_status"] == "llm_verified"


def test_supported_relation_with_non_source_evidence_fails_closed(monkeypatch):
    def invented_quote(values):
        return {"decisions": [decision(values[0], evidence="Alpha definitely produces Beta.")]}

    result = run_extract(monkeypatch, ScriptedVerifier(extraction(), [invented_quote]))
    assert result["relations"][0]["verification_status"] == "verification_failed"
    assert result["_diagnostics"]["published"] == 0
    assert result["_verification_events"][0]["error_category"] == "invalid_decision"
    assert "exact source quote" in result["_verification_events"][0]["reason"]


def test_contradicted_and_insufficient_relations_keep_reason_but_do_not_publish(monkeypatch):
    extracted = extraction([
        {"source": "e1", "target": "e2", "type": "produces", "description": "",
         "evidence": "Alpha produces Beta.", "confidence": 0.9},
        {"source": "e2", "target": "e3", "type": "uses", "description": "",
         "evidence": "Beta uses Gamma.", "confidence": 0.9},
    ])

    def verdicts(values):
        return {"decisions": [
            decision(values[0], "contradicted", reason="text says the opposite"),
            decision(values[1], "insufficient_evidence", reason="predicate is not proven"),
        ]}

    result = run_extract(monkeypatch, ScriptedVerifier(extracted, [verdicts]))
    assert [item["verification_status"] for item in result["relations"]] == ["rejected", "uncertain"]
    assert [item["verification_reason"] for item in result["relations"]] == [
        "text says the opposite", "predicate is not proven",
    ]
    assert result["_diagnostics"]["quarantined"] == 2


def test_reversed_orientation_swaps_endpoints_and_preserves_original_candidate(monkeypatch):
    source = "Alpha is located in Shanghai."
    extracted = {
        "entities": [
            {"id": "e1", "name": "Alpha", "type": "Organization", "description": "", "aliases": [],
             "evidence": "Alpha", "confidence": 0.9},
            {"id": "e2", "name": "Shanghai", "type": "Location", "description": "", "aliases": [],
             "evidence": "Shanghai", "confidence": 0.9},
        ],
        "relations": [{"source": "e2", "target": "e1", "type": "located_in", "description": "",
                       "evidence": source, "confidence": 0.9}],
    }

    def reverse(values):
        return {"decisions": [decision(
            values[0], relation_type="located_in", orientation="reversed", evidence=source,
        )]}

    result = run_extract(monkeypatch, ScriptedVerifier(extracted, [reverse]), source=source)
    relation = result["relations"][0]
    assert (relation["source"], relation["target"]) == ("e1", "e2")
    assert relation["_original_candidate"]["source"] == "e2"
    assert relation["orientation"] == "reversed"
    assert relation["verification_status"] == "llm_verified"
    assert relation["description"] == "Alpha located_in Shanghai"


def test_cooccurrence_and_high_self_reported_confidence_are_not_rules_verified(monkeypatch):
    monkeypatch.setattr(Config, "extraction_gleaning_enabled", False)
    monkeypatch.setattr(Config, "extraction_verification_enabled", False)
    extracted = extraction([{
        "source": "e1", "target": "e2", "type": "produces", "description": "",
        "evidence": "Alpha and Beta", "confidence": 1.0,
    }])
    result = SchemaEvolutionService(None, ScriptedVerifier(extracted, [])).extract_chunk(
        {"id": 4, "project_key": "p4", "llm_model": "test"},
        {"version": 1, "schema_json": BASE_SCHEMA},
        {"chunk_key": "c", "content": "Alpha and Beta are discussed."},
    )
    assert result["relations"][0]["verification_status"] == "uncertain"
    assert result["_diagnostics"]["rules_verified"] == 0


def test_incompatible_predicate_is_not_silently_changed_to_related_to(monkeypatch):
    monkeypatch.setattr(Config, "extraction_gleaning_enabled", False)
    monkeypatch.setattr(Config, "extraction_verification_enabled", False)
    extracted = {
        "entities": [
            {"id": "e1", "name": "Shanghai", "type": "Location", "description": "", "aliases": [],
             "evidence": "Shanghai", "confidence": 0.9},
            {"id": "e2", "name": "Alpha", "type": "Organization", "description": "", "aliases": [],
             "evidence": "Alpha", "confidence": 0.9},
        ],
        "relations": [{"source": "e1", "target": "e2", "type": "located_in", "description": "",
                       "evidence": "Shanghai and Alpha", "confidence": 0.9}],
    }
    result = SchemaEvolutionService(None, ScriptedVerifier(extracted, [])).extract_chunk(
        {"id": 4, "project_key": "p4", "llm_model": "test"},
        {"version": 1, "schema_json": BASE_SCHEMA},
        {"chunk_key": "c", "content": "Shanghai and Alpha are discussed."},
    )
    assert result["relations"][0]["type"] == "located_in"
    assert result["relations"][0]["verification_status"] == "uncertain"


def test_synthetic_chunk_header_is_not_extractable_but_body_evidence_is_located(monkeypatch):
    monkeypatch.setattr(Config, "extraction_gleaning_enabled", False)
    monkeypatch.setattr(Config, "extraction_verification_enabled", False)
    extracted = {
        "entities": [
            {"id": "e1", "name": "secret-file.pdf", "type": "Product", "description": "", "aliases": [],
             "evidence": "secret-file.pdf", "confidence": 0.9},
            {"id": "e2", "name": "Alpha", "type": "Organization", "description": "", "aliases": [],
             "evidence": "Alpha", "confidence": 0.9},
        ],
        "relations": [],
    }
    chunk = {
        "chunk_key": "s1_p001",
        "content": "# Intro\n\n章节ID: s1\n来源文档: secret-file.pdf\n\n分块: 1/1\n\nAlpha builds products.",
        "metadata": {"section_key": "s1", "section_title": "Intro"},
    }
    result = run_extract(
        monkeypatch, ScriptedVerifier(extracted, []), extracted_chunk=chunk, verification=False,
    )
    assert [item["name"] for item in result["entities"]] == ["Alpha"]
    entity = result["entities"][0]
    assert chunk["content"][entity["evidence_start"]:entity["evidence_end"]] == "Alpha"
    assert "secret-file.pdf" not in result["_diagnostics"]


def test_publishable_statuses_are_an_explicit_allowlist():
    assert PUBLISHABLE_RELATION_STATUSES == {"rules_verified", "llm_verified", "manual_verified"}
    for status in ("evidence_unverified", "verification_missing", "verification_failed", "uncertain", "rejected"):
        assert not is_publishable_relation(status)


def test_full_chunk_offsets_cover_entities_relations_blank_lines_and_natural_marker(monkeypatch):
    body = "\n\nAlpha produces Beta.\n分块: this is ordinary source text."
    full_content = (
        "# Intro\n\n章节ID: s1\n来源文档: source.pdf\n\n分块: 1/1\n\n" + body
    )
    chunk = {
        "id": 9,
        "chunk_key": "s1_p001",
        "content": full_content,
        "metadata": {"section_key": "s1", "section_title": "Intro"},
    }
    result = run_extract(
        monkeypatch,
        ScriptedVerifier(extraction(), [lambda values: {"decisions": [decision(values[0])]}]),
        extracted_chunk=chunk,
    )
    for item in [*result["entities"], *result["relations"]]:
        assert full_content[item["evidence_start"]:item["evidence_end"]] == item["evidence_text"]
    original = result["relations"][0]["_original_candidate"]
    assert full_content[original["evidence_start"]:original["evidence_end"]] == original["evidence_text"]
    assert "分块: this is ordinary source text." in result["relations"][0]["evidence_text"] or full_content.endswith(
        "分块: this is ordinary source text."
    )


def test_duplicate_exact_quote_uses_match_nearest_candidate_offset():
    content = "Alpha produces Beta. filler Alpha produces Beta."
    second = content.rfind("Alpha produces Beta.")
    evidence, start, end = SchemaEvolutionService._locate_evidence(
        content, "Alpha produces Beta.", near=second,
    )
    assert (evidence, start, end) == (
        "Alpha produces Beta.", second, second + len("Alpha produces Beta."),
    )


def test_fictitious_name_with_common_alias_cannot_publish_relation(monkeypatch):
    extracted = {
        "entities": [
            {"id": "e1", "name": "Invented Corporation", "type": "Organization",
             "description": "", "aliases": ["company"], "evidence": "company", "confidence": 0.9},
            {"id": "e2", "name": "Beta", "type": "Product", "description": "",
             "aliases": [], "evidence": "Beta", "confidence": 0.9},
        ],
        "relations": [{"source": "e1", "target": "e2", "type": "produces",
                       "description": "", "evidence": "The company produces Beta.", "confidence": 0.9}],
    }
    result = run_extract(
        monkeypatch,
        ScriptedVerifier(extracted, []),
        source="The company produces Beta.",
        verification=False,
    )
    assert [item["name"] for item in result["entities"]] == ["Beta"]
    assert result["relations"] == []


@pytest.mark.parametrize(
    ("name", "source"),
    [("AI", "The claim was rejected."), ("US", "The service uses encryption.")],
)
def test_short_ascii_entity_does_not_match_inside_another_word(monkeypatch, name, source):
    extracted = {
        "entities": [{"id": "e1", "name": name, "type": "GenericEntity",
                      "description": "", "aliases": [], "evidence": name, "confidence": 0.9}],
        "relations": [],
    }
    result = run_extract(
        monkeypatch, ScriptedVerifier(extracted, []), source=source, verification=False,
    )
    assert result["entities"] == []


def test_multiword_ascii_entity_does_not_match_inside_larger_words(monkeypatch):
    extracted = {
        "entities": [{"id": "e1", "name": "US Bank", "type": "Organization",
                      "description": "", "aliases": [], "evidence": "US Bank", "confidence": 0.9}],
        "relations": [],
    }
    result = run_extract(
        monkeypatch,
        ScriptedVerifier(extracted, []),
        source="BUS Banking provides services.",
        verification=False,
    )
    assert result["entities"] == []


def test_supported_as_is_relation_replaces_unverified_description(monkeypatch):
    extracted = extraction([{
        "source": "e1", "target": "e2", "type": "produces",
        "description": "Alpha secretly acquired Gamma.",
        "evidence": "Alpha produces Beta.", "confidence": 0.9,
    }])
    result = run_extract(
        monkeypatch,
        ScriptedVerifier(extracted, [lambda values: {"decisions": [decision(values[0])]}]),
    )
    relation = result["relations"][0]
    assert relation["verification_status"] == "llm_verified"
    assert relation["orientation"] == "as_is"
    assert relation["type"] == "produces"
    assert relation["description"] == "Alpha produces Beta"


@pytest.mark.parametrize(
    ("verdict", "status"),
    [("contradicted", "rejected"), ("insufficient_evidence", "uncertain")],
)
def test_unknown_proposed_type_can_be_rejected_without_schema_error(monkeypatch, verdict, status):
    extracted = extraction([{
        "source": "e1", "target": "e2", "type": "unknown_predicate",
        "description": "bad proposal", "evidence": "Alpha produces Beta.", "confidence": 0.9,
    }])

    def reject(values):
        return {"decisions": [decision(values[0], verdict, reason="source does not support it")]}

    result = run_extract(monkeypatch, ScriptedVerifier(extracted, [reject]))
    assert result["relations"][0]["verification_status"] == status
    assert result["relations"][0]["verification_reason"] == "source does not support it"
