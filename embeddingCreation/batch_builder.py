#!/usr/bin/env python3
"""
Step 6: sentence batching from sentence metadata.

Builds fixed-size batches from metadata rows and preserves input order.
Optionally applies a char_len budget to reduce padding proxy per batch.
ID generation is handled outside this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator, Iterable, Sequence

from packed_id import SentenceMeta

DEFAULT_BATCH_SIZE: int = 64

LOGGER = logging.getLogger("batch_builder")


@dataclass(frozen=True, slots=True)
class SentenceBatch:
    metas: list[SentenceMeta]
    sentences: list[str]


def batch_sentences(
    sentence_metas: Iterable[SentenceMeta],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    char_budget: int | None = None,
) -> Generator[SentenceBatch, None, None]:
    """Accumulate sentence metadata into fixed-size batches.

    Args:
        sentence_metas: Iterable of SentenceMeta values.
        batch_size: Max sentences per batch (must be >= 1).
        char_budget: Optional cap on max_char_len * batch_count (must be >= 1).

    Yields:
        SentenceBatch with metas and text values of equal length.
        Final partial batch is yielded when input is exhausted.

    Raises:
        ValueError: if batch_size < 1, char_budget < 1, or item is not SentenceMeta.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if char_budget is not None and char_budget < 1:
        raise ValueError(f"char_budget must be >= 1 when provided, got {char_budget}")

    batch_metas: list[SentenceMeta] = []
    batch_texts: list[str] = []
    batch_max_char_len = 0
    yielded_batches = 0
    total_sentences = 0
    partial_flush = False

    try:
        for meta in sentence_metas:
            if not isinstance(meta, SentenceMeta):
                raise ValueError("all items must be SentenceMeta")

            if char_budget is not None and batch_texts:
                next_max_char_len = max(batch_max_char_len, meta.char_len)
                next_count = len(batch_texts) + 1
                if next_max_char_len * next_count > char_budget:
                    yielded_batches += 1
                    LOGGER.info(
                        "yielding char-budget batch flush: size=%d max_char_len=%d char_work=%d budget=%d",
                        len(batch_texts),
                        batch_max_char_len,
                        batch_max_char_len * len(batch_texts),
                        char_budget,
                    )
                    yield SentenceBatch(metas=batch_metas, sentences=batch_texts)
                    batch_metas = []
                    batch_texts = []
                    batch_max_char_len = 0

            batch_metas.append(meta)
            batch_texts.append(meta.text)
            batch_max_char_len = max(batch_max_char_len, meta.char_len)
            total_sentences += 1

            if len(batch_texts) == batch_size:
                yielded_batches += 1
                LOGGER.info(
                    "yielding full batch: size=%d max_char_len=%d",
                    len(batch_texts),
                    batch_max_char_len,
                )
                yield SentenceBatch(metas=batch_metas, sentences=batch_texts)
                batch_metas = []
                batch_texts = []
                batch_max_char_len = 0

        if batch_texts:
            yielded_batches += 1
            partial_flush = True
            LOGGER.info(
                "yielding partial batch flush: size=%d max_char_len=%d",
                len(batch_texts),
                batch_max_char_len,
            )
            yield SentenceBatch(metas=batch_metas, sentences=batch_texts)
    finally:
        LOGGER.info(
            "batch summary: batches=%d total_sentences=%d batch_size=%d char_budget=%s partial_flush=%s",
            yielded_batches,
            total_sentences,
            batch_size,
            char_budget,
            partial_flush,
        )


def batch_sentences_dynamic_cap(
    sentence_metas: Sequence[SentenceMeta],
    *,
    batch_size_ceiling: int = DEFAULT_BATCH_SIZE,
    char_budget: int,
) -> Generator[SentenceBatch, None, None]:
    """Build dynamic-size batches using a per-batch char_len budget.

    Assumes input is sorted by non-decreasing char_len. The emitted batch size k is the
    largest value in [1, min(batch_size_ceiling, remaining)] satisfying:
      max_char_len_in_batch * k <= char_budget
    If no k satisfies the inequality, a single-sentence batch is emitted.
    """
    if batch_size_ceiling < 1:
        raise ValueError(f"batch_size_ceiling must be >= 1, got {batch_size_ceiling}")
    if char_budget < 1:
        raise ValueError(f"char_budget must be >= 1, got {char_budget}")

    n = len(sentence_metas)
    for meta in sentence_metas:
        if not isinstance(meta, SentenceMeta):
            raise ValueError("all items must be SentenceMeta")

    yielded_batches = 0
    total_sentences = 0
    i = 0
    while i < n:
        hi = min(batch_size_ceiling, n - i)
        best_k = 1
        for k in range(1, hi + 1):
            max_char_len = sentence_metas[i + k - 1].char_len
            if max_char_len * k <= char_budget:
                best_k = k
            else:
                break

        batch_metas = list(sentence_metas[i : i + best_k])
        batch_texts = [m.text for m in batch_metas]
        yielded_batches += 1
        total_sentences += len(batch_texts)
        LOGGER.info(
            "yielding dynamic-cap batch: size=%d max_char_len=%d char_work=%d budget=%d ceiling=%d",
            best_k,
            batch_metas[-1].char_len,
            batch_metas[-1].char_len * best_k,
            char_budget,
            batch_size_ceiling,
        )
        yield SentenceBatch(metas=batch_metas, sentences=batch_texts)
        i += best_k

    LOGGER.info(
        "dynamic-cap batch summary: batches=%d total_sentences=%d ceiling=%d budget=%d",
        yielded_batches,
        total_sentences,
        batch_size_ceiling,
        char_budget,
    )
