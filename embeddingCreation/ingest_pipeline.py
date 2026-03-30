#!/usr/bin/env python3
"""
Step 11: end-to-end ingestion pipeline orchestrator.

Wires:
file_reader -> sentence_splitter -> batch_builder -> embedding_worker -> binary_writer
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from batch_builder import batch_sentences
from binary_reader import VerificationReport, verify_file
from binary_writer import BinaryWriterError, EmbeddingWriter
from embedding_worker import EmbeddingError, EmbeddingWorker, ModelLoadError
from file_reader import DEFAULT_MAX_FILE_SIZE, discover_files, read_text_files

LOGGER = logging.getLogger("ingest_pipeline")


@dataclass(frozen=True)
class PipelineResult:
    input_directory: str
    output_path: str
    files_discovered: int
    files_read: int
    files_skipped: int
    total_sentences: int
    total_batches: int
    records_written: int
    verification: VerificationReport | None
    elapsed_seconds: float
    success: bool
    errors: list[str]


def _split_text(text: str, *, locale: str) -> list[str]:
    # Lazy import allows unit tests to patch this function without requiring PyICU.
    from sentence_splitter import split_sentences

    return split_sentences(text, locale=locale)


def _validate_startup(in_dir: Path, out_path: Path, *, batch_size: int) -> str | None:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not in_dir.exists():
        return f"[STARTUP] input directory not found: {in_dir}"
    if not in_dir.is_dir():
        return f"[STARTUP] not a directory: {in_dir}"
    if out_path.exists() and out_path.is_dir():
        return f"[STARTUP] output path is a directory: {out_path}"

    parent = out_path.parent
    if not parent.exists():
        return f"[STARTUP] output parent directory does not exist: {parent}"
    if parent.is_file():
        return f"[STARTUP] output parent is not a directory: {parent}"

    try:
        import torch
    except ImportError:
        return "[CUDA] PyTorch is not installed"

    if not torch.cuda.is_available():
        return "[CUDA] CUDA is not available; GPU inference required"
    return None


def _sentence_stream(
    input_directory: str | Path,
    *,
    extensions: frozenset[str],
    recursive: bool,
    encoding: str,
    skip_hidden: bool,
    max_file_size: int,
    locale: str,
    counters: dict[str, int],
    splitter: Callable[[str], list[str]],
):
    for path, text in read_text_files(
        input_directory,
        extensions=extensions,
        recursive=recursive,
        encoding=encoding,
        skip_hidden=skip_hidden,
        max_file_size_bytes=max_file_size,
    ):
        counters["files_read"] += 1
        try:
            sentences = splitter(text)
        except Exception as exc:
            counters["files_skipped"] += 1
            LOGGER.warning("[SPLITTER] skipping file %s: %s", path.name, exc)
            continue
        counters["total_sentences"] += len(sentences)
        LOGGER.info("split file: path=%s sentences=%d", path.name, len(sentences))
        for sentence in sentences:
            yield sentence


def _failure_result(
    *,
    in_dir: Path,
    out_path: Path,
    files_discovered: int,
    files_read: int,
    files_skipped: int,
    total_sentences: int,
    total_batches: int,
    records_written: int,
    errors: list[str],
    started_at: float,
) -> PipelineResult:
    return PipelineResult(
        input_directory=str(in_dir),
        output_path=str(out_path),
        files_discovered=files_discovered,
        files_read=files_read,
        files_skipped=files_skipped,
        total_sentences=total_sentences,
        total_batches=total_batches,
        records_written=records_written,
        verification=None,
        elapsed_seconds=time.perf_counter() - started_at,
        success=False,
        errors=errors,
    )


def run_pipeline(
    input_directory: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 8,
    locale: str = "en_US",
    extensions: frozenset[str] = frozenset({".txt"}),
    recursive: bool = False,
    encoding: str = "utf-8",
    skip_hidden: bool = True,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> PipelineResult:
    t0 = time.perf_counter()
    in_dir = Path(input_directory)
    out_path = Path(output_path)
    startup_error = _validate_startup(in_dir, out_path, batch_size=batch_size)
    if startup_error is not None:
        LOGGER.error(startup_error)
        return _failure_result(
            in_dir=in_dir,
            out_path=out_path,
            files_discovered=0,
            files_read=0,
            files_skipped=0,
            total_sentences=0,
            total_batches=0,
            records_written=0,
            errors=[startup_error],
            started_at=t0,
        )

    errors: list[str] = []
    try:
        files = discover_files(
            in_dir,
            extensions=extensions,
            recursive=recursive,
            skip_hidden=skip_hidden,
            max_file_size_bytes=max_file_size,
        )
    except Exception as exc:
        msg = f"[STARTUP] failed to discover files: {exc}"
        LOGGER.error(msg)
        return _failure_result(
            in_dir=in_dir,
            out_path=out_path,
            files_discovered=0,
            files_read=0,
            files_skipped=0,
            total_sentences=0,
            total_batches=0,
            records_written=0,
            errors=[msg],
            started_at=t0,
        )
    files_discovered = len(files)
    LOGGER.info("files discovered: count=%d directory=%s", files_discovered, in_dir)

    counters: dict[str, int] = {"files_read": 0, "files_skipped": 0, "total_sentences": 0}
    total_batches = 0
    records_written = 0
    success = True
    worker = EmbeddingWorker()

    splitter = lambda text: _split_text(text, locale=locale)
    sentence_iter = _sentence_stream(
        in_dir,
        extensions=extensions,
        recursive=recursive,
        encoding=encoding,
        skip_hidden=skip_hidden,
        max_file_size=max_file_size,
        locale=locale,
        counters=counters,
        splitter=splitter,
    )

    try:
        worker.init()
        with EmbeddingWriter(out_path) as writer:
            for batch in batch_sentences(sentence_iter, batch_size=batch_size, start_id=0):
                total_batches += 1
                embeddings = worker.encode_batch(batch.sentences)
                writer.write_batch(batch.ids, embeddings)
                LOGGER.info(
                    "batch processed: index=%d size=%d records_written=%d",
                    total_batches,
                    len(batch.sentences),
                    writer.records_written,
                )
            records_written = writer.records_written
    except ModelLoadError as exc:
        success = False
        msg = f"[MODEL_LOAD] {exc}"
        errors.append(msg)
        LOGGER.error(msg)
    except EmbeddingError as exc:
        success = False
        msg = f"[ENCODE] {exc}"
        errors.append(msg)
        LOGGER.error(msg)
    except BinaryWriterError as exc:
        success = False
        msg = f"[WRITE] {exc}"
        errors.append(msg)
        LOGGER.error(msg)
    except OverflowError as exc:
        success = False
        msg = f"[OVERFLOW] {exc}"
        errors.append(msg)
        LOGGER.error(msg)
    finally:
        try:
            worker.shutdown()
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.error("worker shutdown failed: %s", exc)

    verification: VerificationReport | None = None
    if success:
        try:
            verification = verify_file(out_path)
            if not verification.ok:
                success = False
                errors.extend(verification.errors)
                LOGGER.error("post-run verification failed for: %s", out_path)
        except Exception as exc:
            success = False
            msg = f"post-run verification error: {exc}"
            errors.append(msg)
            LOGGER.error(msg)

    files_skipped = counters["files_skipped"] + max(0, files_discovered - counters["files_read"])

    elapsed = time.perf_counter() - t0
    result = PipelineResult(
        input_directory=str(in_dir),
        output_path=str(out_path),
        files_discovered=files_discovered,
        files_read=counters["files_read"],
        files_skipped=files_skipped,
        total_sentences=counters["total_sentences"],
        total_batches=total_batches,
        records_written=records_written,
        verification=verification,
        elapsed_seconds=elapsed,
        success=success,
        errors=errors,
    )
    LOGGER.info(
        "pipeline complete: files_discovered=%d files_read=%d sentences=%d batches=%d records=%d elapsed_s=%.4f ok=%s",
        result.files_discovered,
        result.files_read,
        result.total_sentences,
        result.total_batches,
        result.records_written,
        result.elapsed_seconds,
        result.success,
    )
    return result
