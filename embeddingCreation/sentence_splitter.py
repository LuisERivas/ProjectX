#!/usr/bin/env python3
"""
Step 4: ICU-based sentence splitting for the embedding pipeline.

Splits raw document text into ordered, trimmed, non-empty sentences using
ICU BreakIterator with post-merge abbreviation suppression.
Conforms to embedding_format_spec.txt Section 3 contract.
"""

from __future__ import annotations

import icu

SUPPRESSIONS: frozenset[str] = frozenset({
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.",
    "vs.", "etc.", "approx.",
    "e.g.", "i.e.", "vol.", "dept.",
    "Gen.", "Gov.", "Sgt.", "Cpl.", "Pvt.",
    "St.", "Ave.", "Blvd.",
    "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec.",
    "U.S.", "U.K.",
    "No.", "Fig.", "Ref.",
    "a.m.", "p.m.",
})


def _ends_with_suppression(sentence: str) -> bool:
    """True if the sentence is or ends with a suppressed abbreviation."""
    for abbr in SUPPRESSIONS:
        if sentence == abbr or sentence.endswith(" " + abbr):
            return True
    return False


def split_sentences(
    text: str,
    *,
    locale: str = "en_US",
) -> list[str]:
    """Split text into sentences using ICU BreakIterator.

    - Merges fragments that end with suppressed abbreviations (Dr., Mr., etc.)
      back into the following sentence to prevent false breaks.
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

    boundaries: list[int] = list(bi)

    sentences: list[str] = []
    start = 0
    i = 0
    while i < len(boundaries):
        end = boundaries[i]
        chunk = text[start:end].strip()
        while _ends_with_suppression(chunk) and i + 1 < len(boundaries):
            i += 1
            end = boundaries[i]
            chunk = text[start:end].strip()
        if chunk:
            sentences.append(chunk)
        start = end
        i += 1
    return sentences
