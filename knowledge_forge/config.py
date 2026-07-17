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
    service_host = os.getenv("SERVICE_HOST", "127.0.0.1")
    upload_host = os.getenv("UPLOAD_HOST", "0.0.0.0")
    upload_port = int(os.getenv("UPLOAD_PORT", "8011"))
    http_base_url = os.getenv("HTTP_BASE_URL", f"http://{service_host}:8010")

    postgres = {
        "host": os.getenv("POSTGRES_HOST", service_host),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "change-me"),
        "dbname": os.getenv("POSTGRES_DB", "knowledge_forge"),
        "connect_timeout": int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5")),
    }
    postgres_maintenance_db = os.getenv("POSTGRES_MAINTENANCE_DB", "postgres")

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", f"http://{service_host}:11434")
    ollama_keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "-1m")
    ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "480"))
    default_llm_model = os.getenv("DEFAULT_LLM_MODEL", "qwen3.6:latest")
    default_embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "bge-m3:latest")

    hugegraph_host = os.getenv("HUGEGRAPH_HOST", service_host)
    hugegraph_port = int(os.getenv("HUGEGRAPH_PORT", "8080"))
    hugegraph_graph = os.getenv("HUGEGRAPH_GRAPH", "knowledge_forge")
    hugegraph_user = os.getenv("HUGEGRAPH_USER", "admin")
    hugegraph_password = os.getenv("HUGEGRAPH_PASSWORD", os.getenv("HUGEGRAPH_PWD", "admin"))
    hugegraph_timeout = int(os.getenv("HUGEGRAPH_TIMEOUT", "30"))
    hugegraph_page_size = max(1, min(1000, int(os.getenv("HUGEGRAPH_PAGE_SIZE", "500"))))
    hugegraph_hubble_url = os.getenv("HUGEGRAPH_HUBBLE_URL", f"http://{service_host}:18088")
    hugegraph_hubble_graph_host = os.getenv("HUGEGRAPH_HUBBLE_GRAPH_HOST", service_host)

    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "8"))
    retrieval_dense_weight = float(os.getenv("RETRIEVAL_DENSE_WEIGHT", "0.72"))
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
    indexing_failure_retention_days = max(1, int(os.getenv("INDEXING_FAILURE_RETENTION_DAYS", "30")))
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
    schema_review_batch_size = int(os.getenv("SCHEMA_REVIEW_BATCH_SIZE", "4"))
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
        128, int(os.getenv("RELATION_CLASSIFIER_OUTPUT_RESERVE", "512")),
    )
    relation_classifier_safety_margin = max(
        128, int(os.getenv("RELATION_CLASSIFIER_SAFETY_MARGIN", "512")),
    )
    relation_classifier_max_candidates = max(
        1, int(os.getenv("RELATION_CLASSIFIER_MAX_CANDIDATES", "32")),
    )
    relation_classifier_gleaning_window = max(
        1, min(3, int(os.getenv("RELATION_CLASSIFIER_GLEANING_WINDOW", "2"))),
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
