#!/usr/bin/env python3
"""
Step 4: ICU-based sentence splitting for the embedding pipeline.

Splits raw document text into ordered, trimmed, non-empty sentences using
ICU BreakIterator with abbreviation suppression via FilteredBreakIteratorBuilder.
Conforms to embedding_format_spec.txt Section 3 contract.
"""

from __future__ import annotations

import logging

import icu

LOGGER = logging.getLogger("sentence_splitter")

SUPPRESSIONS: tuple[str, ...] = (
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.",
    "vs.", "etc.", "approx.",
    "e.g.", "i.e.", "vol.", "dept.",
    "Gen.", "Gov.", "Sgt.", "Cpl.", "Pvt.",
    "St.", "Ave.", "Blvd.",
    "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec.",
    "U.S.", "U.K.",
    "No.", "Fig.", "Ref.",
    "a.m.", "p.m.",
)

_USE_FILTERED: bool | None = None


def _create_sentence_iterator(locale: str) -> icu.BreakIterator:
    """Build a sentence BreakIterator with abbreviation suppression.

    Falls back to a plain BreakIterator if FilteredBreakIteratorBuilder
    is not available in the installed PyICU version.
    """
    global _USE_FILTERED

    loc = icu.Locale(locale)
    bi = icu.BreakIterator.createSentenceInstance(loc)

    if _USE_FILTERED is False:
        return bi

    try:
        builder = icu.FilteredBreakIteratorBuilder.createInstance(loc)
        for abbr in SUPPRESSIONS:
            builder.suppressBreakAfter(abbr)
        filtered = builder.build(bi)
        if _USE_FILTERED is None:
            _USE_FILTERED = True
            LOGGER.info("using FilteredBreakIteratorBuilder with %d suppressions", len(SUPPRESSIONS))
        return filtered
    except (AttributeError, TypeError, icu.ICUError) as exc:
        if _USE_FILTERED is None:
            _USE_FILTERED = False
            LOGGER.warning(
                "FilteredBreakIteratorBuilder not available; "
                "falling back to plain BreakIterator: %s",
                exc,
            )
        return bi


def split_sentences(
    text: str,
    *,
    locale: str = "en_US",
) -> list[str]:
    """Split text into sentences using ICU BreakIterator.

    - Suppresses false breaks after common abbreviations (Dr., Mr., etc.).
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

    bi = _create_sentence_iterator(locale)
    bi.setText(text)

    sentences: list[str] = []
    start = 0
    for end in bi:
        chunk = text[start:end].strip()
        if chunk:
            sentences.append(chunk)
        start = end
    return sentences
