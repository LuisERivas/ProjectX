#!/usr/bin/env python3
"""
Step 3 embedding worker: load voyage-4-nano once and serve encode batches.

Design constraints:
- CUDA is required (no CPU fallback).
- SentenceTransformer model loads once at init and stays resident.
- Every encode call uses plain encode(...), normalize_embeddings=True.
- Embeddings are validated in float32, then cast to float16 for output/storage.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from embedding_validator import EmbeddingValidationError, validate_embeddings

MODEL_ID = "voyageai/voyage-4-nano"
EXPECTED_DIM = 2048
PRE_CAST_NORM_MIN = 0.999
PRE_CAST_NORM_MAX = 1.001
POST_CAST_NORM_MIN = 0.95
POST_CAST_NORM_MAX = 1.05
FULL_VALIDATION_BATCHES = 3
DEBUG_ENV_VAR = "EMBEDDING_WORKER_DEBUG"

LOGGER = logging.getLogger("embedding_worker")


class WorkerState(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    READY = "READY"
    SHUTDOWN = "SHUTDOWN"


class EmbeddingWorkerError(Exception):
    """Base error for all embedding worker failures."""


class WorkerStateError(EmbeddingWorkerError):
    """Raised when an operation is invalid for the current worker state."""


class ModelLoadError(EmbeddingWorkerError):
    """Raised when the embedding model cannot be loaded correctly."""


class EmbeddingError(EmbeddingWorkerError):
    """Raised for encoding, validation, or output conversion failures."""


@dataclass
class _WorkerMetrics:
    total_batches: int = 0
    total_sentences: int = 0
    total_encode_time_s: float = 0.0
    load_count: int = 0
    init_time_s: float | None = None
    dtype: str | None = None
    device: str | None = None


class EmbeddingWorker:
    """Long-lived in-process embedding worker for Step 3."""

    def __init__(self) -> None:
        self._state = WorkerState.UNINITIALIZED
        self._model: Any | None = None
        self._metrics = _WorkerMetrics()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "model_id": MODEL_ID,
            "device": self._metrics.device,
            "dtype": self._metrics.dtype,
            "init_time_s": self._metrics.init_time_s,
            "total_batches": self._metrics.total_batches,
            "total_sentences": self._metrics.total_sentences,
            "total_encode_time_s": self._metrics.total_encode_time_s,
            "load_count": self._metrics.load_count,
        }

    def init(self) -> None:
        """Load the model once and transition to READY."""
        if self._state == WorkerState.READY:
            raise WorkerStateError("model already loaded; init() called more than once")
        if self._state == WorkerState.SHUTDOWN:
            raise WorkerStateError("worker already shut down; create a new worker instance")

        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise ModelLoadError("required runtime packages for model loading are missing") from exc

        if not torch.cuda.is_available():
            raise ModelLoadError("CUDA not available - this worker requires GPU; no CPU fallback")

        t0 = time.perf_counter()
        last_err: Exception | None = None
        dtype_attempts = [torch.bfloat16, torch.float16]
        for dtype in dtype_attempts:
            try:
                self._model = SentenceTransformer(
                    MODEL_ID,
                    trust_remote_code=True,
                    truncate_dim=EXPECTED_DIM,
                    model_kwargs={
                        "attn_implementation": "sdpa",
                        "torch_dtype": dtype,
                    },
                )
                self._metrics.dtype = str(dtype).replace("torch.", "")
                break
            except Exception as exc:
                last_err = exc
                if dtype is torch.bfloat16:
                    LOGGER.warning("bfloat16 init failed; retrying with float16: %s", exc)
                    continue
                raise ModelLoadError("failed to load SentenceTransformer on CUDA") from exc

        if self._model is None:
            raise ModelLoadError("failed to construct model instance") from last_err

        device = str(getattr(self._model, "device", ""))
        if "cuda" not in device:
            raise ModelLoadError(
                f"model loaded on unexpected device '{device}'; CUDA is required"
            )
        self._metrics.device = device

        # Verify expected embedding dimension once at startup.
        probe = self._model.encode(
            "dimension check",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(probe, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[0]
        if arr.shape[-1] != EXPECTED_DIM:
            raise ModelLoadError(
                f"unexpected embedding dim during init: got {arr.shape[-1]}, expected {EXPECTED_DIM}"
            )

        self._metrics.init_time_s = time.perf_counter() - t0
        self._metrics.load_count += 1
        self._state = WorkerState.READY
        LOGGER.info(
            "embedding worker initialized",
            extra={
                "model_id": MODEL_ID,
                "device": self._metrics.device,
                "dtype": self._metrics.dtype,
                "load_time_s": round(self._metrics.init_time_s, 4),
                "load_count": self._metrics.load_count,
            },
        )

    def encode_batch(self, sentences: list[str]) -> np.ndarray:
        """Encode a sentence batch and return (N, 2048) float16 vectors."""
        if self._state != WorkerState.READY or self._model is None:
            raise WorkerStateError("worker is not READY; call init() before encode_batch()")
        if not isinstance(sentences, list):
            raise ValueError("sentences must be a list[str]")
        if not sentences:
            raise ValueError("sentences must not be empty")
        if not all(isinstance(s, str) for s in sentences):
            raise ValueError("all items in sentences must be str")

        if os.environ.get(DEBUG_ENV_VAR) == "1":
            preview = [s[:100] for s in sentences]
            LOGGER.info("encoding batch debug preview", extra={"batch_preview": preview})

        t0 = time.perf_counter()
        try:
            out = self._model.encode(
                sentences,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            result = np.asarray(out).astype(np.float32, copy=False)
            if result.ndim == 1:
                result = result.reshape(1, -1)

            try:
                skip_norm_checks = self._metrics.total_batches >= FULL_VALIDATION_BATCHES
                result_fp16 = validate_embeddings(
                    result,
                    expected_count=len(sentences),
                    expected_dim=EXPECTED_DIM,
                    pre_cast_norm_min=PRE_CAST_NORM_MIN,
                    pre_cast_norm_max=PRE_CAST_NORM_MAX,
                    post_cast_norm_min=POST_CAST_NORM_MIN,
                    post_cast_norm_max=POST_CAST_NORM_MAX,
                    skip_norm_checks=skip_norm_checks,
                )
            except EmbeddingValidationError as exc:
                raise EmbeddingError(str(exc)) from exc

            elapsed = time.perf_counter() - t0
            self._metrics.total_batches += 1
            self._metrics.total_sentences += len(sentences)
            self._metrics.total_encode_time_s += elapsed

            LOGGER.info(
                "batch encoded",
                extra={
                    "batch_size": len(sentences),
                    "encode_time_s": round(elapsed, 4),
                    "sentences_per_sec": round(len(sentences) / elapsed, 2) if elapsed > 0 else None,
                },
            )
            return result_fp16

        except EmbeddingWorkerError:
            raise
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda error" in msg:
                raise EmbeddingError(
                    f"CUDA encode failure (likely OOM). reduce batch size. original: {exc}"
                ) from exc
            raise EmbeddingError(f"runtime encode failure: {exc}") from exc
        except Exception as exc:
            raise EmbeddingError(f"encode failure: {exc}") from exc

    def shutdown(self) -> None:
        """Unload model and release CUDA cache. Idempotent."""
        if self._state == WorkerState.SHUTDOWN:
            LOGGER.info("shutdown called on already shut down worker")
            return
        if self._state == WorkerState.UNINITIALIZED:
            LOGGER.info("shutdown called on uninitialized worker")
            self._state = WorkerState.SHUTDOWN
            return

        try:
            import torch
        except Exception:
            torch = None

        self._model = None
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                LOGGER.warning("torch.cuda.empty_cache failed during shutdown: %s", exc)

        self._state = WorkerState.SHUTDOWN
        LOGGER.info("model unloaded and worker shut down")

