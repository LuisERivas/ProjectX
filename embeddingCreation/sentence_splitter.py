#!/usr/bin/env python3
"""
Step 4: ICU-based sentence splitting for the embedding pipeline.

Splits raw document text into ordered, trimmed, non-empty sentences using
ICU BreakIterator. Conforms to embedding_format_spec.txt Section 3 contract.
"""

from __future__ import annotations

import icu


def split_sentences(
    text: str,
    *,
    locale: str = "en_US",
) -> list[str]:
    """Split text into sentences using ICU BreakIterator.

    - Normalizes line endings (\\r\\n, \\r -> \\n).
    - Strips whitespace from each sentence.
    - Drops empty strings after stripping.
    - Preserves sentence order from source text.

    Args:
        text: Raw document text (UTF-8 str).
        locale: ICU locale for sentence rules (default "en_US").

    Returns:
        Ordered list of non-empty trimmed sentences.

    Raises:
        ValueError: if text is not a str.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a str")
    if not text or not text.strip():
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    bi = icu.BreakIterator.createSentenceInstance(icu.Locale(locale))
    bi.setText(text)

    sentences: list[str] = []
    start = 0
    for end in bi:
        chunk = text[start:end].strip()
        if chunk:
            sentences.append(chunk)
        start = end
    return sentences
