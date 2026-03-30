#!/usr/bin/env python3
"""
Paragraph splitting helpers for ingest metadata assignment.
"""

from __future__ import annotations


def split_into_paragraphs(text: str) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("text must be a str")
    if not text:
        return []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in normalized.split("\n")]
    return [p for p in paragraphs if p]
