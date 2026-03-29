#!/usr/bin/env python3
"""
Step 5: text file discovery and reading for ingestion.

This module discovers text files from a directory and yields UTF-8 decoded
text content as (path, text) tuples ready for sentence splitting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

DEFAULT_EXTENSIONS: frozenset[str] = frozenset({".txt"})
DEFAULT_MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB

LOGGER = logging.getLogger("file_reader")


def _normalize_extensions(extensions: frozenset[str]) -> frozenset[str]:
    """Normalize extension allowlist to lowercase, dotted suffixes."""
    out: set[str] = set()
    for ext in extensions:
        e = ext.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = f".{e}"
        out.add(e)
    return frozenset(out)


def _is_hidden(path: Path, root: Path) -> bool:
    """Return True if the path or any relative parent segment is hidden."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return any(part.startswith(".") for part in rel.parts)


def discover_files(
    directory: str | Path,
    *,
    extensions: frozenset[str] = DEFAULT_EXTENSIONS,
    recursive: bool = False,
    skip_hidden: bool = True,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE,
) -> list[Path]:
    """Return sorted list of text file paths in directory.

    - Filters by extension (case-insensitive).
    - Skips hidden files/dirs when skip_hidden=True.
    - Skips zero-byte files.
    - Skips files exceeding max_file_size_bytes.
    - Non-recursive by default.
    - Sorted by resolved absolute path for deterministic ordering.
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    allowed = _normalize_extensions(extensions)
    candidates = root.rglob("*") if recursive else root.iterdir()

    discovered: list[Path] = []
    candidate_count = 0
    skipped_count = 0
    for path in candidates:
        candidate_count += 1
        if skip_hidden and _is_hidden(path, root):
            skipped_count += 1
            LOGGER.debug("skipping hidden path: %s", path)
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            skipped_count += 1
            LOGGER.debug("skipping unsupported extension: %s", path)
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            skipped_count += 1
            LOGGER.warning("skipping unreadable file (stat failed): %s (%s)", path, exc)
            continue
        if size == 0:
            skipped_count += 1
            LOGGER.warning("skipping zero-byte file: %s", path)
            continue
        if size > max_file_size_bytes:
            skipped_count += 1
            LOGGER.warning(
                "skipping oversized file: %s (size=%d > max=%d)",
                path,
                size,
                max_file_size_bytes,
            )
            continue
        discovered.append(path)

    discovered.sort(key=lambda p: str(p.resolve()))
    LOGGER.info(
        "discovered files summary: discovered=%d candidates=%d skipped=%d directory=%s recursive=%s",
        len(discovered),
        candidate_count,
        skipped_count,
        root,
        recursive,
    )
    return discovered


def read_text_files(
    directory: str | Path,
    *,
    extensions: frozenset[str] = DEFAULT_EXTENSIONS,
    recursive: bool = False,
    encoding: str = "utf-8",
    skip_hidden: bool = True,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE,
) -> Generator[tuple[Path, str], None, None]:
    """Discover and read text files, yielding (path, text) tuples."""
    paths = discover_files(
        directory,
        extensions=extensions,
        recursive=recursive,
        skip_hidden=skip_hidden,
        max_file_size_bytes=max_file_size_bytes,
    )
    read_count = 0
    skipped_count = 0
    for path in paths:
        try:
            text = path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            skipped_count += 1
            LOGGER.warning("skipping non-%s file: %s (%s)", encoding, path, exc)
            continue
        except OSError as exc:
            skipped_count += 1
            LOGGER.warning("skipping unreadable file: %s (%s)", path, exc)
            continue
        read_count += 1
        LOGGER.info("read file: %s chars=%d", path, len(text))
        yield path, text

    LOGGER.info(
        "file read summary: read=%d skipped=%d discovered=%d directory=%s",
        read_count,
        skipped_count,
        len(paths),
        Path(directory),
    )
