"""Runtime configuration for Knowledge Forge."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Load a small local .env file without replacing exported variables."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


class Config:
    runtime_mode = os.getenv("RUNTIME_MODE", "development").strip().lower()
    service_host = os.getenv("SERVICE_HOST", "127.0.0.1")
    upload_host = os.getenv("UPLOAD_HOST", "127.0.0.1")
    upload_port = int(os.getenv("UPLOAD_PORT", "8011"))
    http_base_url = os.getenv("HTTP_BASE_URL", f"http://{service_host}:8010")
    allow_unsafe_public_bind = os.getenv("ALLOW_UNSAFE_PUBLIC_BIND", "false").lower() == "true"
    public_bind_approved = os.getenv("PUBLIC_BIND_APPROVED", "false").lower() == "true"
    authentication_enabled = os.getenv("AUTHENTICATION_ENABLED", "false").lower() == "true"

    postgres = {
        "host": os.getenv("POSTGRES_HOST", service_host),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
        "dbname": os.getenv("POSTGRES_DB", "knowledge_forge"),
        "connect_timeout": int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5")),
    }
    postgres_maintenance_db = os.getenv("POSTGRES_MAINTENANCE_DB", "postgres")

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", f"http://{service_host}:11434")
    # Embeddings may remain on a separately validated runtime while structured
    # generation uses another accelerator.  The default preserves the original
    # single-endpoint deployment exactly.
    ollama_embedding_base_url = os.getenv(
        "OLLAMA_EMBEDDING_BASE_URL", ollama_base_url,
    )
    # A model tag does not fully identify its numerical vector space: moving
    # the same model between execution backends (for example CPU -> CUDA) can
    # introduce small floating-point differences. Persist a stable execution
    # identity with rebuildable vector indexes. Entity facts themselves remain
    # reusable when the model and extraction policy are unchanged; treating a
    # hardware move as a semantic-input change would force expensive LLM work.
    # If a deployment changes the backend behind the same URL, it must bump
    # the explicit profile value.
    ollama_embedding_runtime_profile = (
        os.getenv("OLLAMA_EMBEDDING_RUNTIME_PROFILE", "").strip()
        or "ollama-endpoint-"
        + hashlib.sha256(ollama_embedding_base_url.encode("utf-8")).hexdigest()[:12]
    )
    ollama_keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "-1m")
    ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "480"))
    # Keep model-backed extraction reproducible across retries and service
    # restarts.  This is an Ollama generation option, not a Python process
    # random seed; changing it intentionally invalidates extraction snapshots.
    ollama_seed = int(os.getenv("OLLAMA_SEED", "17"))
    # Application request budgets. These never alter Ollama's runner num_ctx.
    ollama_effective_context_fallback = int(
        os.getenv("OLLAMA_EFFECTIVE_CONTEXT_FALLBACK_TOKENS", "8192")
    )
    ollama_default_context_budget = int(
        os.getenv("OLLAMA_DEFAULT_CONTEXT_BUDGET_TOKENS", "8192")
    )
    ollama_default_max_output = int(os.getenv("OLLAMA_DEFAULT_MAX_OUTPUT_TOKENS", "2048"))
    ollama_default_safety_margin = int(os.getenv("OLLAMA_SAFETY_MARGIN_TOKENS", "256"))
    ollama_token_variance_absolute = int(
        os.getenv("OLLAMA_TOKEN_VARIANCE_ABSOLUTE_TOKENS", "128")
    )
    ollama_token_variance_ratio = float(os.getenv("OLLAMA_TOKEN_VARIANCE_RATIO", "0.50"))
    ollama_context_cache_seconds = int(os.getenv("OLLAMA_CONTEXT_CACHE_SECONDS", "60"))
    ollama_image_token_estimate = int(os.getenv("OLLAMA_IMAGE_TOKEN_ESTIMATE", "1024"))
    entity_review_max_output_tokens = int(os.getenv("ENTITY_REVIEW_MAX_OUTPUT_TOKENS", "512"))
    relation_review_max_output_tokens = int(os.getenv("RELATION_REVIEW_MAX_OUTPUT_TOKENS", "2048"))
    schema_review_max_output_tokens = int(os.getenv("SCHEMA_REVIEW_MAX_OUTPUT_TOKENS", "8192"))
    # Open relation observations follow the broad schema review.  Recurring,
    # source-grounded observations are grouped into candidate relation
    # families; every family still passes the ordinary evidence, policy,
    # semantic-novelty and independent-verifier gates.
    schema_relation_coverage_rounds = max(
        0, min(2, int(os.getenv("SCHEMA_RELATION_COVERAGE_ROUNDS", "2")))
    )
    schema_relation_compound_audit_enabled = os.getenv(
        "SCHEMA_RELATION_COMPOUND_AUDIT_ENABLED", "true",
    ).strip().lower() in {"1", "true", "yes", "on"}
    schema_relation_induction_batch_size = max(
        1, min(8, int(os.getenv(
            "SCHEMA_RELATION_INDUCTION_BATCH_SIZE", "8",
        )))
    )
    # A single, explicit source-grounded observation can be enough to discover
    # a reusable relation family when a project has only one document.  The
    # candidate still passes evidence, policy, novelty and verifier gates; this
    # switch only controls whether the bounded induction request is scheduled.
    schema_single_observation_relation_induction = os.getenv(
        "SCHEMA_SINGLE_OBSERVATION_RELATION_INDUCTION", "true",
    ).strip().lower() in {"1", "true", "yes", "on"}
    extraction_max_output_tokens = int(os.getenv("EXTRACTION_MAX_OUTPUT_TOKENS", "4096"))
    entity_only_max_output_tokens = int(
        os.getenv("ENTITY_ONLY_MAX_OUTPUT_TOKENS", "8192")
    )
    entity_only_segment_max_chars = max(
        256, int(os.getenv("ENTITY_ONLY_SEGMENT_MAX_CHARS", "900"))
    )
    entity_only_max_sentences_per_segment = max(
        1, min(8, int(os.getenv("ENTITY_ONLY_MAX_SENTENCES_PER_SEGMENT", "2")))
    )
    entity_only_max_items_per_segment = max(
        8, int(os.getenv("ENTITY_ONLY_MAX_ITEMS_PER_SEGMENT", "48"))
    )
    # The enhanced path uses one compact mention inventory instead of the
    # former primary-entity + proposition-argument + endpoint-gleaning stack.
    # It deliberately has its own segmentation budget so legacy extraction
    # fixtures and replayed snapshots retain their immutable protocol.
    entity_grounded_inventory_enabled = os.getenv(
        "ENTITY_GROUNDED_INVENTORY_ENABLED", "true",
    ).strip().lower() in {"1", "true", "yes", "on"}
    entity_grounded_segment_max_chars = max(
        512, int(os.getenv("ENTITY_GROUNDED_SEGMENT_MAX_CHARS", "2400"))
    )
    entity_grounded_max_sentences_per_segment = max(
        1, min(24, int(os.getenv(
            "ENTITY_GROUNDED_MAX_SENTENCES_PER_SEGMENT", "8",
        )))
    )
    entity_grounded_max_items_per_segment = max(
        16, min(256, int(os.getenv(
            "ENTITY_GROUNDED_MAX_ITEMS_PER_SEGMENT", "96",
        )))
    )
    entity_grounded_gleaning_rounds = max(
        0, min(1, int(os.getenv(
            "ENTITY_GROUNDED_GLEANING_ROUNDS", "1",
        )))
    )
    entity_endpoint_gleaning_rounds = max(
        0, min(2, int(os.getenv("ENTITY_ENDPOINT_GLEANING_ROUNDS", "1")))
    )
    image_analysis_max_output_tokens = int(os.getenv("IMAGE_ANALYSIS_MAX_OUTPUT_TOKENS", "2048"))
    default_llm_model = os.getenv("DEFAULT_LLM_MODEL", "qwen3.6:latest")
    default_embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "bge-m3:latest")
    # Native GraphRAG Standard indexing is an isolated long-running job.  The
    # budget is execution control only; it never changes GraphRAG algorithms or
    # permits publication of an incomplete staging bundle.
    graphrag_native_timeout_seconds = max(
        600, int(os.getenv("GRAPHRAG_NATIVE_TIMEOUT_SECONDS", "7200"))
    )

    hugegraph_host = os.getenv("HUGEGRAPH_HOST", service_host)
    hugegraph_port = int(os.getenv("HUGEGRAPH_PORT", "8080"))
    hugegraph_graph = os.getenv("HUGEGRAPH_GRAPH", "knowledge_forge")
    hugegraph_auth_enabled = os.getenv("HUGEGRAPH_AUTH_ENABLED", "false").lower() == "true"
    hugegraph_user = os.getenv("HUGEGRAPH_USER", "")
    hugegraph_password = os.getenv("HUGEGRAPH_PASSWORD", os.getenv("HUGEGRAPH_PWD", ""))
    hugegraph_timeout = int(os.getenv("HUGEGRAPH_TIMEOUT", "30"))
    hugegraph_page_size = max(1, min(1000, int(os.getenv("HUGEGRAPH_PAGE_SIZE", "500"))))
    graph_lease_seconds = max(30, int(os.getenv("GRAPH_LEASE_SECONDS", "900")))
    hugegraph_hubble_url = os.getenv("HUGEGRAPH_HUBBLE_URL", f"http://{service_host}:18088")
    hugegraph_hubble_graph_host = os.getenv("HUGEGRAPH_HUBBLE_GRAPH_HOST", service_host)

    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "8"))
    retrieval_dense_weight = float(os.getenv("RETRIEVAL_DENSE_WEIGHT", "0.72"))
    # Detailed community reports are an asynchronous enhancement over the
    # immutable Knowledge Forge graph.  Graph publication and ordinary
    # chunk/entity/relation retrieval never wait for these model calls.
    knowledge_forge_community_reports_enabled = os.getenv(
        "KF_COMMUNITY_REPORTS_ENABLED", "true",
    ).strip().lower() in {"1", "true", "yes", "on"}
    knowledge_forge_community_reports_auto_build = os.getenv(
        "KF_COMMUNITY_REPORTS_AUTO_BUILD", "true",
    ).strip().lower() in {"1", "true", "yes", "on"}
    # Large hierarchical graphs can contain hundreds of report-worthy
    # communities.  Generate a bounded, priority-ordered initial set and leave
    # the rest available for exact on-demand activation from graph browsing or
    # retrieval, rather than monopolising the local LLM for many hours.
    knowledge_forge_community_report_initial_limit = max(
        1, min(200, int(os.getenv("KF_COMMUNITY_REPORT_INITIAL_LIMIT", "12"))),
    )
    knowledge_forge_community_report_search_activation_limit = max(
        1, min(10, int(os.getenv(
            "KF_COMMUNITY_REPORT_SEARCH_ACTIVATION_LIMIT", "2",
        ))),
    )
    knowledge_forge_community_report_min_entities = max(
        2, int(os.getenv("KF_COMMUNITY_REPORT_MIN_ENTITIES", "2")),
    )
    knowledge_forge_community_report_min_relations = max(
        1, int(os.getenv("KF_COMMUNITY_REPORT_MIN_RELATIONS", "1")),
    )
    knowledge_forge_community_report_max_entities = max(
        4, min(80, int(os.getenv("KF_COMMUNITY_REPORT_MAX_ENTITIES", "24"))),
    )
    knowledge_forge_community_report_max_relations = max(
        4, min(120, int(os.getenv("KF_COMMUNITY_REPORT_MAX_RELATIONS", "36"))),
    )
    knowledge_forge_community_report_context_tokens = max(
        2048, int(os.getenv("KF_COMMUNITY_REPORT_CONTEXT_TOKENS", "8192")),
    )
    knowledge_forge_community_report_max_output_tokens = max(
        512, int(os.getenv("KF_COMMUNITY_REPORT_MAX_OUTPUT_TOKENS", "1800")),
    )
    knowledge_forge_community_report_max_attempts = max(
        1, min(5, int(os.getenv("KF_COMMUNITY_REPORT_MAX_ATTEMPTS", "2"))),
    )
    knowledge_forge_community_report_search_top_k = max(
        1, min(20, int(os.getenv("KF_COMMUNITY_REPORT_SEARCH_TOP_K", "5"))),
    )
    pgvector_candidate_multiplier = int(os.getenv("PGVECTOR_CANDIDATE_MULTIPLIER", "8"))
    pgvector_max_candidates = int(os.getenv("PGVECTOR_MAX_CANDIDATES", "200"))
    pgvector_hnsw_m = int(os.getenv("PGVECTOR_HNSW_M", "16"))
    pgvector_hnsw_ef_construction = int(os.getenv("PGVECTOR_HNSW_EF_CONSTRUCTION", "64"))
    pgvector_hnsw_ef_search = int(os.getenv("PGVECTOR_HNSW_EF_SEARCH", "100"))
    pgvector_hnsw_max_dimensions = int(os.getenv("PGVECTOR_HNSW_MAX_DIMENSIONS", "2000"))
    entity_embedding_batch_size = max(1, int(os.getenv("ENTITY_EMBEDDING_BATCH_SIZE", "16")))
    indexing_retry_batch_size = max(1, int(os.getenv("INDEXING_RETRY_BATCH_SIZE", "100")))
    indexing_retry_max_attempts = max(1, int(os.getenv("INDEXING_RETRY_MAX_ATTEMPTS", "8")))
    indexing_retry_base_seconds = max(1, int(os.getenv("INDEXING_RETRY_BASE_SECONDS", "30")))
    indexing_failure_cleanup_enabled = os.getenv(
        "INDEXING_FAILURE_CLEANUP_ENABLED", "false",
    ).lower() == "true"
    indexing_failure_retention_days = max(1, int(os.getenv("INDEXING_FAILURE_RETENTION_DAYS", "30")))
    audit_redaction_enabled = os.getenv("AUDIT_REDACTION_ENABLED", "true").lower() == "true"
    audit_retention_days = max(1, int(os.getenv("AUDIT_RETENTION_DAYS", "90")))
    audit_max_rows = max(1000, int(os.getenv("AUDIT_MAX_ROWS", "100000")))
    audit_cleanup_enabled = os.getenv("AUDIT_CLEANUP_ENABLED", "true").lower() == "true"
    entity_merge_threshold = float(os.getenv("ENTITY_MERGE_THRESHOLD", "0.90"))
    entity_lexical_top_k = int(os.getenv("ENTITY_LEXICAL_TOP_K", "10"))
    entity_vector_top_k = int(os.getenv("ENTITY_VECTOR_TOP_K", "20"))
    entity_lexical_max_k = max(
        entity_lexical_top_k, int(os.getenv("ENTITY_LEXICAL_MAX_K", "50"))
    )
    entity_vector_max_k = max(
        entity_vector_top_k, int(os.getenv("ENTITY_VECTOR_MAX_K", "100"))
    )
    entity_lexical_recall_threshold = float(os.getenv("ENTITY_LEXICAL_RECALL_THRESHOLD", "0.45"))
    entity_vector_recall_threshold = float(os.getenv("ENTITY_VECTOR_RECALL_THRESHOLD", "0.70"))
    entity_llm_review_threshold = float(os.getenv("ENTITY_LLM_REVIEW_THRESHOLD", "0.68"))
    entity_llm_accept_threshold = float(os.getenv("ENTITY_LLM_ACCEPT_THRESHOLD", "0.85"))
    entity_llm_max_candidates = int(os.getenv("ENTITY_LLM_MAX_CANDIDATES", "5"))
    entity_llm_review_enabled = os.getenv("ENTITY_LLM_REVIEW_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }
    entity_auto_merge_lexical_threshold = float(os.getenv("ENTITY_AUTO_MERGE_LEXICAL_THRESHOLD", "0.96"))
    entity_auto_merge_vector_threshold = float(os.getenv("ENTITY_AUTO_MERGE_VECTOR_THRESHOLD", "0.82"))
    entity_reranker_enabled = os.getenv("ENTITY_RERANKER_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }
    # Entity resolution consumes reranking over an internal HTTP boundary.  The
    # model process has its own lifecycle so the web/job process never imports
    # torch or owns model memory.
    entity_reranker_url = os.getenv(
        "ENTITY_RERANKER_URL", "http://127.0.0.1:8012",
    ).strip().rstrip("/")
    entity_reranker_model = os.getenv(
        "ENTITY_RERANKER_MODEL", "bge-reranker-v2-m3",
    ).strip()
    entity_reranker_api_key = os.getenv("RERANKER_API_KEY", "").strip()
    entity_reranker_connect_timeout = max(
        1, int(os.getenv("ENTITY_RERANKER_CONNECT_TIMEOUT", "5")),
    )
    entity_reranker_timeout = max(
        entity_reranker_connect_timeout,
        int(os.getenv("ENTITY_RERANKER_TIMEOUT", "240")),
    )
    entity_reranker_retries = max(
        0, min(5, int(os.getenv("ENTITY_RERANKER_RETRIES", "2"))),
    )

    # Dedicated reranker service settings.  They are intentionally separate
    # from ENTITY_RERANKER_URL so this process can be deployed on another host.
    reranker_host = os.getenv("RERANKER_HOST", "127.0.0.1").strip()
    reranker_port = int(os.getenv("RERANKER_PORT", "8012"))
    reranker_model_name = os.getenv(
        "RERANKER_MODEL_NAME", entity_reranker_model,
    ).strip()
    reranker_allow_unauthenticated_lan = os.getenv(
        "RERANKER_ALLOW_UNAUTHENTICATED_LAN", "false",
    ).strip().lower() in {"1", "true", "yes", "on"}
    reranker_preload = os.getenv(
        "RERANKER_PRELOAD", "true",
    ).strip().lower() in {"1", "true", "yes", "on"}
    reranker_max_documents = max(
        1, min(1024, int(os.getenv("RERANKER_MAX_DOCUMENTS", "128"))),
    )
    reranker_max_text_chars = max(
        256, int(os.getenv("RERANKER_MAX_TEXT_CHARS", "20000")),
    )
    reranker_max_total_chars = max(
        reranker_max_text_chars,
        int(os.getenv("RERANKER_MAX_TOTAL_CHARS", "500000")),
    )
    reranker_max_body_bytes = max(
        4096, int(os.getenv("RERANKER_MAX_BODY_BYTES", "1048576")),
    )
    reranker_max_pending_requests = max(
        1, min(256, int(os.getenv("RERANKER_MAX_PENDING_REQUESTS", "32"))),
    )
    reranker_queue_timeout = max(
        1, int(os.getenv("RERANKER_QUEUE_TIMEOUT", "180")),
    )
    entity_reranker_model_path = os.getenv("ENTITY_RERANKER_MODEL_PATH", "/models/entity-reranker")
    entity_reranker_device = os.getenv("ENTITY_RERANKER_DEVICE", "auto").strip().lower()
    entity_reranker_batch_size = max(1, int(os.getenv("ENTITY_RERANKER_BATCH_SIZE", "4")))
    entity_reranker_max_length = max(64, int(os.getenv("ENTITY_RERANKER_MAX_LENGTH", "384")))
    entity_reranker_top_k = max(1, int(os.getenv("ENTITY_RERANKER_TOP_K", "10")))
    entity_reranker_max_k = max(
        entity_reranker_top_k, int(os.getenv("ENTITY_RERANKER_MAX_K", "30"))
    )
    entity_reranker_gpu_min_free_mb = max(0, int(os.getenv("ENTITY_RERANKER_GPU_MIN_FREE_MB", "4096")))
    entity_reranker_review_threshold = float(os.getenv("ENTITY_RERANKER_REVIEW_THRESHOLD", "0.55"))
    entity_reranker_auto_threshold = float(os.getenv("ENTITY_RERANKER_AUTO_THRESHOLD", "0.93"))
    schema_auto_apply = os.getenv("SCHEMA_AUTO_APPLY", "true").strip().lower() in {"1", "true", "yes", "on"}
    # Ordinary Knowledge Forge builds are recall-first and publish grounded
    # open facts against the currently active logical Schema. Expensive LLM
    # Schema discovery is a separate governance job by default, so it cannot
    # block first graph availability or turn normalization into a recall gate.
    schema_discovery_during_build = os.getenv(
        "SCHEMA_DISCOVERY_DURING_BUILD", "false",
    ).strip().lower() in {"1", "true", "yes", "on"}
    schema_sample_base_chunks = max(
        1,
        int(os.getenv("SCHEMA_SAMPLE_BASE_CHUNKS", os.getenv("SCHEMA_SAMPLE_MAX_CHUNKS", "12"))),
    )
    schema_sample_ceiling_chunks = max(
        schema_sample_base_chunks, int(os.getenv("SCHEMA_SAMPLE_CEILING_CHUNKS", "48"))
    )
    schema_sample_base_chars = max(
        1,
        int(os.getenv("SCHEMA_SAMPLE_BASE_CHARS", os.getenv("SCHEMA_SAMPLE_MAX_CHARS", "12000"))),
    )
    schema_sample_ceiling_chars = max(
        schema_sample_base_chars, int(os.getenv("SCHEMA_SAMPLE_CEILING_CHARS", "60000"))
    )
    schema_sample_chars_per_chunk = max(
        160, int(os.getenv("SCHEMA_SAMPLE_CHARS_PER_CHUNK", "2400"))
    )
    schema_sample_min_chars = int(os.getenv("SCHEMA_SAMPLE_MIN_CHARS", "160"))
    schema_sample_per_section = int(os.getenv("SCHEMA_SAMPLE_PER_SECTION", "2"))
    schema_review_batch_size = int(os.getenv("SCHEMA_REVIEW_BATCH_SIZE", "1"))
    # Project builds review a stratified corpus sample rather than evolving the
    # logical Schema once per document.  Four excerpts fit comfortably in the
    # configured local-model context while cutting long-corpus review calls by
    # roughly 75%; the discovery routine still splits a batch on truncation.
    schema_corpus_review_batch_size = max(
        1, min(8, int(os.getenv("SCHEMA_CORPUS_REVIEW_BATCH_SIZE", "4")))
    )
    schema_review_min_confidence = float(os.getenv("SCHEMA_REVIEW_MIN_CONFIDENCE", "0.85"))
    schema_review_single_evidence_confidence = float(
        os.getenv("SCHEMA_REVIEW_SINGLE_EVIDENCE_CONFIDENCE", "0.96")
    )
    schema_review_min_evidence = int(os.getenv("SCHEMA_REVIEW_MIN_EVIDENCE", "2"))
    schema_review_novelty_threshold = float(os.getenv("SCHEMA_REVIEW_NOVELTY_THRESHOLD", "0.22"))
    schema_max_entity_additions = int(os.getenv("SCHEMA_MAX_ENTITY_ADDITIONS", "5"))
    schema_max_relation_additions = int(os.getenv("SCHEMA_MAX_RELATION_ADDITIONS", "8"))
    extraction_gleaning_enabled = os.getenv("EXTRACTION_GLEANING_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }
    extraction_verification_enabled = os.getenv("EXTRACTION_VERIFICATION_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }
    extraction_rule_accept_confidence = float(os.getenv("EXTRACTION_RULE_ACCEPT_CONFIDENCE", "0.78"))
    extraction_min_relation_confidence = float(os.getenv("EXTRACTION_MIN_RELATION_CONFIDENCE", "0.55"))
    extraction_gleaning_information_score = float(os.getenv("EXTRACTION_GLEANING_INFORMATION_SCORE", "0.35"))
    relation_shadow_enabled = os.getenv("RELATION_SHADOW_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }
    relation_classifier_output_reserve = max(
        128, int(os.getenv("RELATION_CLASSIFIER_OUTPUT_RESERVE", "2048")),
    )
    relation_classifier_safety_margin = max(
        128, int(os.getenv("RELATION_CLASSIFIER_SAFETY_MARGIN", "512")),
    )
    relation_classifier_max_candidates = max(
        1, min(32, int(os.getenv("RELATION_CLASSIFIER_MAX_CANDIDATES", "8"))),
    )
    # Relation prompts must never repeat an entire long document for every
    # candidate.  The source inventory remains complete; pair-level prompts
    # receive a bounded, provenance-preserving display around their endpoints.
    relation_model_context_max_chars = max(
        2048, min(200_000, int(os.getenv("RELATION_MODEL_CONTEXT_MAX_CHARS", "24000"))),
    )
    relation_classifier_context_max_chars = max(
        2048, min(100_000, int(os.getenv("RELATION_CLASSIFIER_CONTEXT_MAX_CHARS", "12000"))),
    )
    relation_grounded_open_facts_enabled = os.getenv(
        "RELATION_GROUNDED_OPEN_FACTS_ENABLED", "true",
    ).strip().lower() in {"1", "true", "yes", "on"}
    relation_open_fact_gleaning_rounds = max(
        0, min(2, int(os.getenv("RELATION_OPEN_FACT_GLEANING_ROUNDS", "1"))),
    )
    # Preserve an exact grounded source relation as a bounded parallel safety
    # hypothesis. It remains through normalization and every publication gate;
    # only a precise logical predicate that survives the final role audit may
    # suppress an exact duplicate at publication. This prevents Schema coverage
    # from becoming a recall gate. It still passes deterministic and independent
    # verification and is never a co-occurrence fallback.
    relation_open_carrier_max_per_context = max(
        0,
        min(
            128,
            int(os.getenv(
                "RELATION_OPEN_CARRIER_MAX_PER_CONTEXT", "96",
            )),
        ),
    )
    # Coreference is a separate semantic decision from open-fact extraction.
    # Run one bounded predicate-blind adjudicator over unresolved local
    # references before deterministic syntax recovery.
    relation_reference_adjudication_enabled = os.getenv(
        "RELATION_REFERENCE_ADJUDICATION_ENABLED", "true",
    ).strip().lower() in {"1", "true", "yes", "on"}
    # Binding recovery is optional rather than a universal second pass.  The
    # frozen six-domain calibration recovered only two extra bindings from six
    # additional calls, while the deterministic semantic shortlist remains the
    # ordinary recall backstop.  Keep the knob for unusually recall-sensitive
    # deployments, but avoid one low-yield model call per context by default.
    relation_source_binding_gleaning_rounds = max(
        0,
        min(
            2,
            int(os.getenv("RELATION_SOURCE_BINDING_GLEANING_ROUNDS", "0")),
        ),
    )
    # A tiny deterministic semantic shortlist is a recall backstop after the
    # enumerative model binder.  These rows remain ordinary hypotheses and
    # cannot publish without validation and the independent verifier.
    relation_source_binding_recall_alternatives = max(
        0,
        min(
            4,
            int(os.getenv(
                "RELATION_SOURCE_BINDING_RECALL_ALTERNATIVES", "1",
            )),
        ),
    )
    relation_source_binding_recall_min_score = max(
        0.0,
        min(
            1.0,
            float(os.getenv(
                "RELATION_SOURCE_BINDING_RECALL_MIN_SCORE", "0.18",
            )),
        ),
    )
    relation_verifier_batch_size = max(
        1,
        min(
            32,
            int(os.getenv(
                "RELATION_VERIFIER_BATCH_SIZE",
                os.getenv("RELATION_SOURCE_VERIFIER_BATCH_SIZE", "8"),
            )),
        ),
    )
    relation_verifier_batch_max_chars = max(
        4096,
        min(
            200_000,
            int(os.getenv(
                "RELATION_VERIFIER_BATCH_MAX_CHARS",
                os.getenv(
                    "RELATION_SOURCE_VERIFIER_BATCH_MAX_CHARS", "48000",
                ),
            )),
        ),
    )
    # Candidate-ledger persistence is database-only operational work.  Keep
    # it deliberately separate from semantic batch sizes so deployment tuning
    # cannot invalidate completed relation snapshots.
    relation_ledger_persist_batch_size = max(
        128,
        min(
            8192,
            int(os.getenv("RELATION_LEDGER_PERSIST_BATCH_SIZE", "4096")),
        ),
    )
    # Batch verification is the efficient primary pass. A small project-run
    # budget independently rechecks only provenance-defined high-risk accepts
    # and high-confidence source-grounded rejects.
    relation_verifier_focused_review_max = max(
        0,
        min(
            256,
            int(os.getenv("RELATION_VERIFIER_FOCUSED_REVIEW_MAX", "24")),
        ),
    )
    relation_verification_mode = os.getenv(
        "RELATION_VERIFICATION_MODE", "lean",
    ).strip().lower()
    if relation_verification_mode not in {"lean", "legacy_consensus"}:
        relation_verification_mode = "lean"
    relation_shadow_max_attempts = max(
        1, min(20, int(os.getenv("RELATION_SHADOW_MAX_ATTEMPTS", "3"))),
    )
    relation_secondary_enabled = os.getenv(
        "RELATION_SECONDARY_ENABLED", "false",
    ).strip().lower() in {"1", "true", "yes", "on"}
    relation_secondary_max_candidates = max(
        1, min(512, int(os.getenv("RELATION_SECONDARY_MAX_CANDIDATES", "64"))),
    )
    relation_proposer_failure_circuit_breaker = max(
        0, min(100, int(os.getenv("RELATION_PROPOSER_FAILURE_CIRCUIT_BREAKER", "0"))),
    )
    relation_proposer_max_proposals = max(
        0, min(128, int(os.getenv("RELATION_PROPOSER_MAX_PROPOSALS", "8"))),
    )
    relation_shadow_lease_seconds = max(
        10, int(os.getenv("RELATION_SHADOW_LEASE_SECONDS", "300")),
    )
    relation_shadow_retry_delay_seconds = max(
        0, int(os.getenv("RELATION_SHADOW_RETRY_DELAY_SECONDS", "0")),
    )
    relation_classifier_gleaning_window = max(
        1, min(3, int(os.getenv("RELATION_CLASSIFIER_GLEANING_WINDOW", "2"))),
    )
    relation_sentence_window = max(
        0, min(1, int(os.getenv("RELATION_SENTENCE_WINDOW", "1"))),
    )
    relation_max_char_distance = max(
        1, min(4096, int(os.getenv("RELATION_MAX_CHAR_DISTANCE", "384"))),
    )
    relation_max_targets_per_source = max(
        1, min(128, int(os.getenv("RELATION_MAX_TARGETS_PER_SOURCE", "12"))),
    )
    relation_max_candidates_per_chunk = max(
        1, min(2048, int(os.getenv("RELATION_MAX_CANDIDATES_PER_CHUNK", "256"))),
    )
    files_root = PROJECT_ROOT / "KnowledgeFiles"
    docs_root = files_root / "docs"
    chunks_root = files_root / "chunks"
    media_root = files_root / "media"
    trash_root = files_root / ".trash"

    image_analysis_enabled = os.getenv("IMAGE_ANALYSIS_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }
    image_ocr_enabled = os.getenv("IMAGE_OCR_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }
    image_min_width = max(1, int(os.getenv("IMAGE_MIN_WIDTH", "96")))
    image_min_height = max(1, int(os.getenv("IMAGE_MIN_HEIGHT", "96")))
    image_min_pixels = max(1, int(os.getenv("IMAGE_MIN_PIXELS", "16000")))
    image_max_dimension = max(256, int(os.getenv("IMAGE_MAX_DIMENSION", "2048")))
    image_scan_text_threshold = max(0, int(os.getenv("IMAGE_SCAN_TEXT_THRESHOLD", "40")))
    image_pdf_render_scale = max(1.0, float(os.getenv("IMAGE_PDF_RENDER_SCALE", "1.7")))

    upload_max_bytes = int(os.getenv("UPLOAD_MAX_MB", "200")) * 1024 * 1024
    chunk_max_chars = int(os.getenv("CHUNK_MAX_CHARS", "2400"))
    chunk_overlap_chars = int(os.getenv("CHUNK_OVERLAP_CHARS", "160"))
    general_chunk_target_tokens = int(os.getenv("GENERAL_CHUNK_TARGET_TOKENS", "512"))
    general_chunk_max_tokens = int(os.getenv("GENERAL_CHUNK_MAX_TOKENS", "800"))
    general_chunk_overlap_percent = int(os.getenv("GENERAL_CHUNK_OVERLAP_PERCENT", "12"))
    general_coalesce_min_sections = max(
        0, int(os.getenv("GENERAL_COALESCE_MIN_SECTIONS", "24"))
    )
    paper_chunk_target_tokens = int(os.getenv("PAPER_CHUNK_TARGET_TOKENS", "640"))
    paper_chunk_max_tokens = int(os.getenv("PAPER_CHUNK_MAX_TOKENS", "900"))
    paper_chunk_overlap_percent = int(os.getenv("PAPER_CHUNK_OVERLAP_PERCENT", "10"))
    pdf_probe_pages = int(os.getenv("PDF_PROBE_PAGES", "5"))
    docling_enabled = os.getenv("DOCLING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    docling_device = os.getenv("DOCLING_DEVICE", "auto").strip().lower()
    docling_cuda_fallback = os.getenv("DOCLING_CUDA_FALLBACK", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }
    docling_gpu_min_free_mb = int(os.getenv("DOCLING_GPU_MIN_FREE_MB", "6144"))
    docling_gpu_memory_fraction = float(os.getenv("DOCLING_GPU_MEMORY_FRACTION", "0.20"))
    docling_batch_size = max(1, int(os.getenv("DOCLING_BATCH_SIZE", "1")))
    docling_num_threads = max(1, int(os.getenv("DOCLING_NUM_THREADS", "4")))

    allowed_suffixes = {".docx", ".pdf", ".md", ".markdown", ".txt", ".xlsx", ".xls", ".csv"}

    @staticmethod
    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def safe_name(value: str, fallback: str = "untitled") -> str:
        value = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", value or fallback)
        return value.strip("._ ") or fallback

    @classmethod
    def project_storage_key(cls, project_id: int, name: str) -> str:
        return f"{cls.safe_name(name, 'project')}_{project_id}"

    @classmethod
    def file_url(cls, path: Path) -> str:
        relative = path.resolve().relative_to(cls.files_root.resolve()).as_posix()
        return f"{cls.http_base_url.rstrip('/')}/{quote(relative, safe='/._-')}"
