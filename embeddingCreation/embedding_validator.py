#!/usr/bin/env python3
"""
Step 7 embedding validation utilities.

Validates raw embedding arrays and converts to float16 for storage.
"""

from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger("embedding_validator")


class EmbeddingValidationError(Exception):
    """Raised when embedding validation fails."""


def validate_embeddings(
    embeddings_fp32: object,
    *,
    expected_count: int,
    expected_dim: int = 2048,
    pre_cast_norm_min: float = 0.999,
    pre_cast_norm_max: float = 1.001,
    post_cast_norm_min: float = 0.95,
    post_cast_norm_max: float = 1.05,
    skip_norm_checks: bool = False,
) -> object:
    """Validate and convert embeddings from fp32 to fp16.

    Returns:
        np.ndarray of shape (expected_count, expected_dim), dtype float16.

    Raises:
        EmbeddingValidationError: on validation or conversion failure.
    """
    try:
        import torch  # type: ignore
    except Exception:
        torch = None

    if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(embeddings_fp32):
        arr_t = embeddings_fp32
        if arr_t.ndim == 1:
            arr_t = arr_t.reshape(1, -1)
        if arr_t.ndim != 2:
            raise EmbeddingValidationError(f"expected 2D embeddings array, got ndim={arr_t.ndim}")
        if arr_t.shape[0] != expected_count:
            raise EmbeddingValidationError(
                f"embedding count mismatch: got {arr_t.shape[0]}, expected {expected_count}"
            )
        if arr_t.shape[1] != expected_dim:
            raise EmbeddingValidationError(
                f"embedding dim mismatch: got {arr_t.shape[1]}, expected {expected_dim}"
            )
        arr_t = arr_t.to(dtype=torch.float32)
        result_fp16_t = arr_t.to(dtype=torch.float16)
        if skip_norm_checks:
            LOGGER.debug(
                "embedding validation fast path (tensor): count=%d dim=%d",
                arr_t.shape[0],
                arr_t.shape[1],
            )
            return result_fp16_t

        if not torch.isfinite(arr_t).all():
            raise EmbeddingValidationError("non-finite values in fp32 embedding output")
        denom = torch.linalg.norm(arr_t, dim=1, keepdim=True)
        if torch.any(~torch.isfinite(denom)) or torch.any(denom <= 0):
            raise EmbeddingValidationError("invalid norms encountered during fp32 normalization")
        arr_t = arr_t / denom

        norms = torch.linalg.norm(arr_t, dim=1)
        if not torch.all((norms >= pre_cast_norm_min) & (norms <= pre_cast_norm_max)):
            raise EmbeddingValidationError(
                f"fp32 norm check failed; expected in [{pre_cast_norm_min}, {pre_cast_norm_max}]"
            )

        result_fp16_t = arr_t.to(dtype=torch.float16)
        post_cast = result_fp16_t.to(dtype=torch.float32)
        if not torch.isfinite(post_cast).all():
            raise EmbeddingValidationError("non-finite values introduced after fp16 cast")
        post_norms = torch.linalg.norm(post_cast, dim=1)
        if not torch.all((post_norms >= post_cast_norm_min) & (post_norms <= post_cast_norm_max)):
            raise EmbeddingValidationError(
                f"fp16 norm check failed; expected in [{post_cast_norm_min}, {post_cast_norm_max}]"
            )

        LOGGER.debug(
            "embedding validation passed (tensor): count=%d dim=%d pre_norm_min=%.6f pre_norm_max=%.6f post_norm_min=%.6f post_norm_max=%.6f",
            arr_t.shape[0],
            arr_t.shape[1],
            float(norms.min().item()) if norms.numel() else float("nan"),
            float(norms.max().item()) if norms.numel() else float("nan"),
            float(post_norms.min().item()) if post_norms.numel() else float("nan"),
            float(post_norms.max().item()) if post_norms.numel() else float("nan"),
        )
        return result_fp16_t

    arr = np.asarray(embeddings_fp32, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise EmbeddingValidationError(f"expected 2D embeddings array, got ndim={arr.ndim}")

    if arr.shape[0] != expected_count:
        raise EmbeddingValidationError(
            f"embedding count mismatch: got {arr.shape[0]}, expected {expected_count}"
        )
    if arr.shape[1] != expected_dim:
        raise EmbeddingValidationError(
            f"embedding dim mismatch: got {arr.shape[1]}, expected {expected_dim}"
        )
    result_fp16 = arr.astype(np.float16)

    if skip_norm_checks:
        LOGGER.debug(
            "embedding validation fast path: count=%d dim=%d",
            arr.shape[0],
            arr.shape[1],
        )
        return result_fp16

    if not np.isfinite(arr).all():
        raise EmbeddingValidationError("non-finite values in fp32 embedding output")

    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(~np.isfinite(denom)) or np.any(denom <= 0):
        raise EmbeddingValidationError("invalid norms encountered during fp32 normalization")
    arr = arr / denom

    norms = np.linalg.norm(arr, axis=1)
    if not np.all((norms >= pre_cast_norm_min) & (norms <= pre_cast_norm_max)):
        raise EmbeddingValidationError(
            f"fp32 norm check failed; expected in [{pre_cast_norm_min}, {pre_cast_norm_max}]"
        )

    result_fp16 = arr.astype(np.float16)
    post_cast = result_fp16.astype(np.float32)
    if not np.isfinite(post_cast).all():
        raise EmbeddingValidationError("non-finite values introduced after fp16 cast")
    post_norms = np.linalg.norm(post_cast, axis=1)
    if not np.all((post_norms >= post_cast_norm_min) & (post_norms <= post_cast_norm_max)):
        raise EmbeddingValidationError(
            f"fp16 norm check failed; expected in [{post_cast_norm_min}, {post_cast_norm_max}]"
        )

    LOGGER.debug(
        "embedding validation passed: count=%d dim=%d pre_norm_min=%.6f pre_norm_max=%.6f post_norm_min=%.6f post_norm_max=%.6f",
        arr.shape[0],
        arr.shape[1],
        float(norms.min()) if norms.size else float("nan"),
        float(norms.max()) if norms.size else float("nan"),
        float(post_norms.min()) if post_norms.size else float("nan"),
        float(post_norms.max()) if post_norms.size else float("nan"),
    )
    return result_fp16
