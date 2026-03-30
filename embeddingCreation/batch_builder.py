#!/usr/bin/env python3
"""
Step 6: sentence batching from sentence metadata.

Builds fixed-size batches from metadata rows and preserves input order.
ID generation is handled outside this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator, Iterable

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
) -> Generator[SentenceBatch, None, None]:
    """Accumulate sentence metadata into fixed-size batches.

    Args:
        sentence_metas: Iterable of SentenceMeta values.
        batch_size: Max sentences per batch (must be >= 1).

    Yields:
        SentenceBatch with metas and text values of equal length.
        Final partial batch is yielded when input is exhausted.

    Raises:
        ValueError: if batch_size < 1 or item is not SentenceMeta.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    batch_metas: list[SentenceMeta] = []
    batch_texts: list[str] = []
    yielded_batches = 0
    total_sentences = 0
    partial_flush = False

    try:
        for meta in sentence_metas:
            if not isinstance(meta, SentenceMeta):
                raise ValueError("all items must be SentenceMeta")

            batch_metas.append(meta)
            batch_texts.append(meta.text)
            total_sentences += 1

            if len(batch_texts) == batch_size:
                yielded_batches += 1
                LOGGER.info(
                    "yielding full batch: size=%d",
                    len(batch_texts),
                )
                yield SentenceBatch(metas=batch_metas, sentences=batch_texts)
                batch_metas = []
                batch_texts = []

        if batch_texts:
            yielded_batches += 1
            partial_flush = True
            LOGGER.info(
                "yielding partial batch flush: size=%d",
                len(batch_texts),
            )
            yield SentenceBatch(metas=batch_metas, sentences=batch_texts)
    finally:
        LOGGER.info(
            "batch summary: batches=%d total_sentences=%d batch_size=%d partial_flush=%s",
            yielded_batches,
            total_sentences,
            batch_size,
            partial_flush,
        )
