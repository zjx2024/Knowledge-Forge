"""Lazy local cross-encoder used as one signal in entity resolution."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from .config import Config
from .entity_identity import entity_identity_text


LOGGER = logging.getLogger(__name__)


class EntityReranker:
    """Score mention/candidate relevance without treating it as identity probability."""

    def __init__(self) -> None:
        self.model_path = Path(Config.entity_reranker_model_path)
        self._lock = threading.RLock()
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device = "cpu"
        self._disabled_reason = ""
        self._last_fallback_reason = ""
        self._last_score_used_model = False

    @property
    def available(self) -> bool:
        return Config.entity_reranker_enabled and self.model_path.exists() and not self._disabled_reason

    @property
    def last_score_used_model(self) -> bool:
        return self._last_score_used_model

    def diagnostics(self) -> dict[str, Any]:
        enabled = bool(Config.entity_reranker_enabled)
        exists = self.model_path.exists()
        return {
            "reranker_enabled": enabled,
            "reranker_loaded": self._model is not None,
            "last_score_used_model": self._last_score_used_model,
            "reranker_model": str(self.model_path),
            "reranker_device": self._device,
            "fallback_reason": (
                self._disabled_reason
                or self._last_fallback_reason
                or ("model path does not exist" if enabled and not exists else "")
                or ("disabled by configuration" if not enabled else "")
            ),
        }

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if not self.available:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                device = "cpu"
                if Config.entity_reranker_device in {"auto", "cuda"} and torch.cuda.is_available():
                    free_bytes, _ = torch.cuda.mem_get_info(0)
                    free_mb = int(free_bytes / 1024 / 1024)
                    if free_mb >= Config.entity_reranker_gpu_min_free_mb:
                        device = "cuda:0"
                    elif Config.entity_reranker_device == "cuda":
                        LOGGER.warning("实体重排器显存不足（%s MiB），回退 CPU", free_mb)
                tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.model_path), local_files_only=True,
                    dtype=torch.float16 if device.startswith("cuda") else torch.float32,
                )
                model.eval().to(device)
                self._torch, self._tokenizer, self._model, self._device = torch, tokenizer, model, device
                self._last_fallback_reason = ""
                LOGGER.info("实体重排器已加载: %s (%s)", self.model_path, device)
                return True
            except Exception as exc:
                self._disabled_reason = str(exc)
                LOGGER.exception("实体重排器加载失败，将使用规则与 LLM 回退: %s", exc)
                return False

    @staticmethod
    def _pair(raw: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
        mention = entity_identity_text(
            raw["name"], raw["type"], raw.get("description") or "", raw.get("aliases") or [],
        )
        canonical = entity_identity_text(
            candidate["name"], candidate["entity_type"], candidate.get("description") or "",
            candidate.get("aliases") or [],
        )
        return [mention, canonical]

    def score(self, raw: dict[str, Any], candidates: list[dict[str, Any]]) -> list[float]:
        self._last_score_used_model = False
        if not candidates or not self._load():
            return [0.0] * len(candidates)
        with self._lock:
            try:
                pairs = [self._pair(raw, candidate) for candidate in candidates]
                scores: list[float] = []
                batch_size = max(1, Config.entity_reranker_batch_size)
                for start in range(0, len(pairs), batch_size):
                    batch = pairs[start : start + batch_size]
                    inputs = self._tokenizer(
                        batch, padding=True, truncation=True,
                        max_length=Config.entity_reranker_max_length, return_tensors="pt",
                    ).to(self._device)
                    with self._torch.inference_mode():
                        logits = self._model(**inputs, return_dict=True).logits.view(-1).float()
                        scores.extend(float(value) for value in self._torch.sigmoid(logits).cpu().tolist())
                self._last_score_used_model = True
                return scores
            except self._torch.cuda.OutOfMemoryError as exc:
                self._torch.cuda.empty_cache()
                self._last_fallback_reason = f"CUDA OOM: {exc}"
                LOGGER.warning("实体重排器 GPU OOM，本批次跳过重排: %s", exc)
                return [0.0] * len(candidates)
            except Exception as exc:
                self._last_fallback_reason = str(exc)
                LOGGER.warning("实体重排失败，本批次使用规则与 LLM 回退: %s", exc)
                return [0.0] * len(candidates)
