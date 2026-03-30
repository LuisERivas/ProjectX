#!/usr/bin/env python3
"""
Step 11: end-to-end ingestion pipeline orchestrator.

Wires:
file_reader -> sentence_splitter -> batch_builder -> embedding_worker -> binary_writer
"""

from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable

from batch_builder import DEFAULT_BATCH_SIZE, batch_sentences, batch_sentences_dynamic_cap
from binary_reader import VerificationReport, verify_file
from binary_writer import BinaryWriterError, EmbeddingWriter
from embedding_worker import EmbeddingError, EmbeddingWorker, ModelLoadError
from file_reader import DEFAULT_MAX_FILE_SIZE, discover_files, read_text_files
from packed_id import (
    DOC_PARA_MAX,
    PARA_LINE_MAX,
    SHARD_DOC_MAX,
    SentenceMeta,
    clamp_char_len,
    pack_id,
)
from paragraph_splitter import split_into_paragraphs

LOGGER = logging.getLogger("ingest_pipeline")
# Full ladder for use with max_probe_batch; default cap 128 preserves prior default probe set.
FULL_PROBE_CANDIDATE_BATCH_SIZES: tuple[int, ...] = (16, 32, 64, 128, 256, 512)
PROBE_CANDIDATE_BATCH_SIZES: tuple[int, ...] = (16, 32, 64, 128)
MAX_PENDING_WRITES: int = 2
CHAR_BUDGET_PER_SENTENCE: int = 256
DEFAULT_PROBE_EPSILON: float = 0.05
CUDA_WARN_FREE_BYTES: int = 1 * 1024 * 1024 * 1024


class ProbeStrategy(str, Enum):
    """How to pick a winner after successful probe encodes."""

    MIN_LATENCY_PER_SENT = "min_latency_per_sent"
    MAX_SUCCESSFUL = "max_successful"
    EPSILON = "epsilon"


def _resolved_probe_candidates(
    probe_batch_sizes: tuple[int, ...] | None,
    max_probe_batch: int | None,
) -> tuple[int, ...]:
    """Ordered distinct positive candidates; default ladder capped at 128 unless max_probe_batch set."""
    if max_probe_batch is not None and max_probe_batch < 1:
        raise ValueError(f"max_probe_batch must be >= 1 when set, got {max_probe_batch}")
    if probe_batch_sizes is not None:
        out = tuple(sorted({c for c in probe_batch_sizes if c > 0}))
        if max_probe_batch is not None:
            out = tuple(c for c in out if c <= max_probe_batch)
        return out
    cap = max_probe_batch if max_probe_batch is not None else 128
    return tuple(c for c in FULL_PROBE_CANDIDATE_BATCH_SIZES if c <= cap)


def _pick_probe_winner(
    attempted: list[tuple[int, float]],
    *,
    strategy: ProbeStrategy,
    probe_epsilon: float,
) -> int | None:
    if not attempted:
        return None
    if strategy is ProbeStrategy.MAX_SUCCESSFUL:
        return max(c for c, _ in attempted)
    if strategy is ProbeStrategy.MIN_LATENCY_PER_SENT:
        best_lat = min(lat for _, lat in attempted)
        for c, lat in attempted:
            if lat == best_lat:
                return c
        return None
    if strategy is ProbeStrategy.EPSILON:
        if probe_epsilon < 0:
            raise ValueError(f"probe_epsilon must be >= 0, got {probe_epsilon}")
        best_lat = min(lat for _, lat in attempted)
        threshold = (1.0 + probe_epsilon) * best_lat
        eligible = [c for c, lat in attempted if lat <= threshold]
        return max(eligible)
    raise ValueError(f"unknown probe strategy: {strategy!r}")


def _maybe_warn_cuda_memory_before_probe(candidate: int) -> None:
    if candidate <= 8:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes < CUDA_WARN_FREE_BYTES:
            LOGGER.warning(
                "probe: low free CUDA memory before batch_size=%d: free=%.2f GB total=%.2f GB",
                candidate,
                free_bytes / (1024**3),
                total_bytes / (1024**3),
            )
    except Exception:
        pass


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


def _file_sentence_stream(
    input_directory: str | Path,
    *,
    extensions: frozenset[str],
    recursive: bool,
    encoding: str,
    skip_hidden: bool,
    max_file_size: int,
    counters: dict[str, int],
    splitter: Callable[[str], list[str]],
) -> Iterable[tuple[int, Path, list[SentenceMeta]]]:
    for shard_doc_num, (path, text) in enumerate(
        read_text_files(
            input_directory,
            extensions=extensions,
            recursive=recursive,
            encoding=encoding,
            skip_hidden=skip_hidden,
            max_file_size_bytes=max_file_size,
        )
    ):
        counters["files_read"] += 1
        try:
            paragraphs = split_into_paragraphs(text)
            metas: list[SentenceMeta] = []
            for doc_para_num, paragraph in enumerate(paragraphs):
                if doc_para_num > DOC_PARA_MAX:
                    raise ValueError(
                        f"doc_para_num overflow in {path.name}: {doc_para_num} > {DOC_PARA_MAX}"
                    )
                para_sentences = splitter(paragraph)
                for para_line, sentence in enumerate(para_sentences):
                    if para_line > PARA_LINE_MAX:
                        raise ValueError(
                            f"para_line overflow in {path.name}: {para_line} > {PARA_LINE_MAX}"
                        )
                    metas.append(
                        SentenceMeta(
                            text=sentence,
                            para_line=para_line,
                            char_len=clamp_char_len(len(sentence)),
                            doc_para_num=doc_para_num,
                            shard_doc_num=shard_doc_num,
                            shard_num=0,
                        )
                    )
        except Exception as exc:
            counters["files_skipped"] += 1
            LOGGER.warning("[SPLITTER] skipping file %s: %s", path.name, exc)
            continue
        counters["total_sentences"] += len(metas)
        LOGGER.info("split file: path=%s sentences=%d", path.name, len(metas))
        if shard_doc_num > SHARD_DOC_MAX:
            raise ValueError(
                f"shard_doc_num overflow for file {path.name}: {shard_doc_num} > {SHARD_DOC_MAX}"
            )
        yield shard_doc_num, path, metas


def probe_batch_size(
    worker: EmbeddingWorker,
    sample_sentences: list[str],
    *,
    candidates: tuple[int, ...] = PROBE_CANDIDATE_BATCH_SIZES,
    fallback_batch_size: int = DEFAULT_BATCH_SIZE,
    strategy: ProbeStrategy = ProbeStrategy.MIN_LATENCY_PER_SENT,
    probe_epsilon: float = DEFAULT_PROBE_EPSILON,
    log_cuda_memory_warn: bool = False,
) -> int:
    if fallback_batch_size < 1:
        raise ValueError(f"fallback_batch_size must be >= 1, got {fallback_batch_size}")
    if not sample_sentences:
        return fallback_batch_size

    attempted: list[tuple[int, float]] = []
    candidate_list = sorted({c for c in candidates if c > 0})
    if not candidate_list:
        return fallback_batch_size

    for candidate in candidate_list:
        if candidate > len(sample_sentences):
            continue
        if log_cuda_memory_warn:
            _maybe_warn_cuda_memory_before_probe(candidate)
        try:
            t0 = time.perf_counter()
            _ = worker.encode_batch(sample_sentences[:candidate])
            elapsed = time.perf_counter() - t0
        except EmbeddingError as exc:
            LOGGER.warning(
                "probe batch failed at size=%d, skipping larger candidates: %s",
                candidate,
                exc,
            )
            break
        latency_per_sentence = elapsed / candidate
        attempted.append((candidate, latency_per_sentence))

    best_candidate = _pick_probe_winner(
        attempted, strategy=strategy, probe_epsilon=probe_epsilon
    )

    if best_candidate is None:
        LOGGER.info(
            "probe had no valid candidates, using fallback_batch_size=%d",
            fallback_batch_size,
        )
        return fallback_batch_size

    LOGGER.info(
        "probe complete: strategy=%s tested=%s winner=%d",
        strategy.value,
        ", ".join(f"{bs}:{lat:.6f}s_per_sent" for bs, lat in attempted),
        best_candidate,
    )
    return best_candidate


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
    batch_size: int = DEFAULT_BATCH_SIZE,
    locale: str = "en_US",
    extensions: frozenset[str] = frozenset({".txt"}),
    recursive: bool = False,
    encoding: str = "utf-8",
    skip_hidden: bool = True,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    probe_batch_sizes: tuple[int, ...] | None = None,
    max_probe_batch: int | None = None,
    probe_strategy: ProbeStrategy | str = ProbeStrategy.MIN_LATENCY_PER_SENT,
    probe_epsilon: float = DEFAULT_PROBE_EPSILON,
    probe_log_cuda_memory: bool = False,
) -> PipelineResult:
    t0 = time.perf_counter()
    in_dir = Path(input_directory)
    out_path = Path(output_path)
    strategy = (
        probe_strategy
        if isinstance(probe_strategy, ProbeStrategy)
        else ProbeStrategy(probe_strategy)
    )
    resolved_probe = _resolved_probe_candidates(probe_batch_sizes, max_probe_batch)
    if not resolved_probe:
        raise ValueError(
            "probe candidate list is empty; adjust probe_batch_sizes or max_probe_batch"
        )
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

    try:
        worker.init()
        with EmbeddingWriter(out_path) as writer:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending_writes: deque[Future[None]] = deque()
                for _, path, file_metas in _file_sentence_stream(
                    in_dir,
                    extensions=extensions,
                    recursive=recursive,
                    encoding=encoding,
                    skip_hidden=skip_hidden,
                    max_file_size=max_file_size,
                    counters=counters,
                    splitter=splitter,
                ):
                    if not file_metas:
                        continue
                    file_metas_sorted = sorted(file_metas, key=lambda m: m.char_len)
                    file_sentences = [m.text for m in file_metas_sorted]
                    probe_window = max(resolved_probe)
                    min_sentences_to_probe = min(resolved_probe)
                    if len(file_sentences) >= min_sentences_to_probe:
                        head_ceiling = probe_batch_size(
                            worker,
                            file_sentences[:probe_window],
                            candidates=resolved_probe,
                            fallback_batch_size=batch_size,
                            strategy=strategy,
                            probe_epsilon=probe_epsilon,
                            log_cuda_memory_warn=probe_log_cuda_memory,
                        )
                        tail_ceiling = probe_batch_size(
                            worker,
                            file_sentences[-probe_window:],
                            candidates=resolved_probe,
                            fallback_batch_size=batch_size,
                            strategy=strategy,
                            probe_epsilon=probe_epsilon,
                            log_cuda_memory_warn=probe_log_cuda_memory,
                        )
                        selected_batch_size = min(head_ceiling, tail_ceiling)
                        LOGGER.info(
                            "file probe ceilings: file=%s head=%d tail=%d final=%d",
                            path.name,
                            head_ceiling,
                            tail_ceiling,
                            selected_batch_size,
                        )
                    else:
                        selected_batch_size = batch_size
                    char_budget = selected_batch_size * CHAR_BUDGET_PER_SENTENCE
                    LOGGER.info(
                        "file batch controls: file=%s sentences=%d batch_size=%d char_budget=%d",
                        path.name,
                        len(file_metas_sorted),
                        selected_batch_size,
                        char_budget,
                    )

                    for batch in batch_sentences_dynamic_cap(
                        file_metas_sorted,
                        batch_size_ceiling=selected_batch_size,
                        char_budget=char_budget,
                    ):
                        total_batches += 1
                        embeddings = worker.encode_batch(batch.sentences)
                        ids = [
                            pack_id(
                                m.para_line,
                                m.char_len,
                                m.doc_para_num,
                                m.shard_num,
                                m.shard_doc_num,
                            )
                            for m in batch.metas
                        ]
                        pending_writes.append(
                            pool.submit(writer.write_batch, ids, embeddings)
                        )
                        if len(pending_writes) >= MAX_PENDING_WRITES:
                            pending_writes.popleft().result()
                        LOGGER.info(
                            "batch queued: index=%d size=%d",
                            total_batches,
                            len(batch.sentences),
                        )
                while pending_writes:
                    pending_writes.popleft().result()
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
    except ValueError as exc:
        success = False
        msg = f"[ID] {exc}"
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
