#!/usr/bin/env python3
"""
Step 6: sentence batching with uint32 ID mapping.

Builds fixed-size batches from an input sentence stream while assigning
monotonic uint32 IDs. Output is ready for EmbeddingWorker.encode_batch().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator, Iterable

DEFAULT_BATCH_SIZE: int = 64
UINT32_MAX: int = (2**32) - 1

LOGGER = logging.getLogger("batch_builder")


@dataclass(frozen=True, slots=True)
class SentenceBatch:
    ids: list[int]
    sentences: list[str]


def batch_sentences(
    sentences: Iterable[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    start_id: int = 0,
) -> Generator[SentenceBatch, None, None]:
    """Accumulate sentences into fixed-size batches with uint32 IDs.

    Args:
        sentences: Iterable of sentence strings.
        batch_size: Max sentences per batch (must be >= 1).
        start_id: First assigned ID (must be >= 0).

    Yields:
        SentenceBatch with ids and sentences of equal length.
        Final partial batch is yielded when input is exhausted.

    Raises:
        ValueError: if batch_size < 1, start_id < 0, or sentence item not str.
        OverflowError: if assigned ID exceeds uint32 max (2^32 - 1).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if start_id < 0:
        raise ValueError(f"start_id must be >= 0, got {start_id}")

    next_id = start_id
    batch_ids: list[int] = []
    batch_texts: list[str] = []
    yielded_batches = 0
    total_sentences = 0
    partial_flush = False

    try:
        for sentence in sentences:
            if not isinstance(sentence, str):
                raise ValueError("all sentence items must be str")
            if next_id > UINT32_MAX:
                raise OverflowError(
                    f"sentence ID {next_id} exceeds uint32 max ({UINT32_MAX})"
                )

            batch_ids.append(next_id)
            batch_texts.append(sentence)
            next_id += 1
            total_sentences += 1

            if len(batch_texts) == batch_size:
                yielded_batches += 1
                LOGGER.info(
                    "yielding full batch: size=%d id_start=%d id_end=%d",
                    len(batch_texts),
                    batch_ids[0],
                    batch_ids[-1],
                )
                yield SentenceBatch(ids=batch_ids, sentences=batch_texts)
                batch_ids = []
                batch_texts = []

        if batch_texts:
            yielded_batches += 1
            partial_flush = True
            LOGGER.info(
                "yielding partial batch flush: size=%d id_start=%d id_end=%d",
                len(batch_texts),
                batch_ids[0],
                batch_ids[-1],
            )
            yield SentenceBatch(ids=batch_ids, sentences=batch_texts)
    finally:
        LOGGER.info(
            "batch summary: batches=%d total_sentences=%d batch_size=%d partial_flush=%s",
            yielded_batches,
            total_sentences,
            batch_size,
            partial_flush,
        )
