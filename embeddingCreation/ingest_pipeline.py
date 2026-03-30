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

import numpy as np

from batch_builder import DEFAULT_BATCH_SIZE, batch_sentences, batch_sentences_dynamic_cap
from binary_reader import VerificationReport, verify_file
from binary_writer import BinaryWriterError, EmbeddingWriter
from embedding_worker import EmbeddingError, EmbeddingWorker, ModelLoadError
from file_reader import DEFAULT_MAX_FILE_SIZE, discover_files, read_text_files
from packed_id import (
    CHAR_LEN_MAX,
    DOC_PARA_MAX,
    PARA_LINE_MAX,
    SHARD_DOC_MAX,
    SentenceMeta,
    clamp_char_len,
    pack_id,
)
from paragraph_splitter import split_into_paragraphs

LOGGER = logging.getLogger("ingest_pipeline")
FULL_PROBE_CANDIDATE_BATCH_SIZES: tuple[int, ...] = (
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
)
PROBE_CANDIDATE_BATCH_SIZES: tuple[int, ...] = (64, 128)
DEFAULT_CHAR_LEN_BUCKET_EDGES: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024)
DEFAULT_MAX_BUCKETS: int = 6
DEFAULT_GVF_THRESHOLD: float = 0.85
JENKS_SAMPLE_SIZE: int = 2000
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
    *,
    char_len_bucketing: bool = False,
) -> tuple[int, ...]:
    """Ordered distinct positive candidates.

    When probe_batch_sizes is None: use FULL_PROBE_CANDIDATE_BATCH_SIZES capped by max_probe_batch,
    or 128 (non-bucketing) / 16384 (bucketing) when max_probe_batch is None.
    """
    if max_probe_batch is not None and max_probe_batch < 1:
        raise ValueError(f"max_probe_batch must be >= 1 when set, got {max_probe_batch}")
    if probe_batch_sizes is not None:
        out = tuple(sorted({c for c in probe_batch_sizes if c > 0}))
        if max_probe_batch is not None:
            out = tuple(c for c in out if c <= max_probe_batch)
        return out
    if max_probe_batch is not None:
        cap = max_probe_batch
    else:
        cap = 16384 if char_len_bucketing else 128
    return tuple(c for c in FULL_PROBE_CANDIDATE_BATCH_SIZES if c <= cap)


def _validate_char_len_bucket_edges(edges: tuple[int, ...]) -> None:
    if not edges:
        raise ValueError("char_len_bucket_edges must be non-empty when bucketing is enabled")
    prev = -1
    for e in edges:
        if e < 1:
            raise ValueError(f"char_len bucket edge must be >= 1, got {e}")
        if e > CHAR_LEN_MAX:
            raise ValueError(
                f"char_len bucket edge must be <= CHAR_LEN_MAX ({CHAR_LEN_MAX}), got {e}"
            )
        if e <= prev:
            raise ValueError(
                f"char_len bucket edges must be strictly increasing, got {edges!r}"
            )
        prev = e


def _split_sorted_metas_by_char_len_edges(
    metas: list[SentenceMeta], edges: tuple[int, ...]
) -> list[list[SentenceMeta]]:
    """Partition metas (sorted by non-decreasing char_len) into non-empty char_len bands."""
    _validate_char_len_bucket_edges(edges)
    n = len(metas)
    if n == 0:
        return []
    result: list[list[SentenceMeta]] = []
    i = 0
    chunk: list[SentenceMeta] = []
    while i < n and metas[i].char_len < edges[0]:
        chunk.append(metas[i])
        i += 1
    if chunk:
        result.append(chunk)
    for j in range(len(edges) - 1):
        chunk = []
        hi = edges[j + 1]
        while i < n and metas[i].char_len < hi:
            chunk.append(metas[i])
            i += 1
        if chunk:
            result.append(chunk)
    chunk = []
    while i < n:
        chunk.append(metas[i])
        i += 1
    if chunk:
        result.append(chunk)
    return result


def _char_len_band_label(char_len: int, edges: tuple[int, ...]) -> str:
    if char_len < edges[0]:
        return f"[0,{edges[0]})"
    if char_len >= edges[-1]:
        return f"[{edges[-1]},inf)"
    for j in range(len(edges) - 1):
        if edges[j] <= char_len < edges[j + 1]:
            return f"[{edges[j]},{edges[j + 1]})"
    return f"[char_len={char_len}]"


def _jenks_breaks(values: list[int], k: int) -> list[int]:
    """Compute Jenks natural breaks for sorted 1D integer data (numpy-accelerated).

    Returns k-1 break points (upper edge of each class except the last).
    values must be sorted non-decreasing and len(values) >= k >= 2.
    """
    n = len(values)
    if k < 2 or n < k:
        return []

    arr = np.asarray(values, dtype=np.float64)
    lower_class_limits = np.zeros((n + 1, k + 1), dtype=np.float64)
    variance_combos = np.full((n + 1, k + 1), np.inf, dtype=np.float64)
    lower_class_limits[1, 1:k + 1] = 1.0
    variance_combos[1, 1:k + 1] = 0.0

    for i in range(2, n + 1):
        sum_ = 0.0
        sum_sq = 0.0
        for m in range(1, i + 1):
            lower_m = i - m + 1
            val = arr[lower_m - 1]
            sum_ += val
            sum_sq += val * val
            variance = sum_sq - (sum_ * sum_) / m
            if lower_m > 1:
                prev = variance_combos[lower_m - 1, 1:k]
                combo = variance + prev
                cur = variance_combos[i, 2:k + 1]
                mask = combo < cur
                if np.any(mask):
                    js = np.where(mask)[0]
                    variance_combos[i, js + 2] = combo[js]
                    lower_class_limits[i, js + 2] = float(lower_m)
        lower_class_limits[i, 1] = 1.0
        variance_combos[i, 1] = variance

    kclass = [0] * (k + 1)
    kclass[k] = n
    kclass[0] = 0
    pivot = k
    while pivot >= 2:
        idx = int(lower_class_limits[kclass[pivot], pivot]) - 1
        kclass[pivot - 1] = idx
        pivot -= 1

    breaks: list[int] = []
    for c in range(1, k):
        edge_idx = kclass[c]
        if 0 <= edge_idx < n:
            breaks.append(int(arr[edge_idx]))
    return breaks


def _goodness_of_variance_fit(values: list[int], breaks: list[int]) -> float:
    """GVF: 1 - (within-class variance / total variance). 1.0 = perfect."""
    if not values:
        return 1.0
    arr = np.asarray(values, dtype=np.float64)
    sdam = float(np.sum((arr - arr.mean()) ** 2))
    if sdam == 0.0:
        return 1.0
    edges = np.array(breaks + [int(arr[-1]) + 1], dtype=np.float64)
    sdcm = 0.0
    lo = 0
    for hi_edge in edges:
        hi_idx = int(np.searchsorted(arr[lo:], hi_edge, side="left")) + lo
        if hi_idx > lo:
            cluster = arr[lo:hi_idx]
            sdcm += float(np.sum((cluster - cluster.mean()) ** 2))
        lo = hi_idx
    return 1.0 - sdcm / sdam


def _sample_sorted(values: list[int], n_sample: int) -> list[int]:
    """Uniformly sample from a sorted list, preserving sort order."""
    n = len(values)
    if n <= n_sample:
        return values
    indices = np.linspace(0, n - 1, n_sample, dtype=np.intp)
    return [values[i] for i in indices]


def _jenks_auto_bucket_edges(
    char_lens: list[int],
    *,
    max_k: int = DEFAULT_MAX_BUCKETS,
    gvf_threshold: float = DEFAULT_GVF_THRESHOLD,
) -> tuple[int, ...]:
    """Find natural bucket edges for sorted char_len array using Jenks + GVF.

    For large inputs (> JENKS_SAMPLE_SIZE), operates on a uniform sample to
    keep runtime bounded, then evaluates GVF against the full dataset.
    Returns a tuple of break-point values (strictly increasing) to be used
    as bucket edges with _split_sorted_metas_by_char_len_edges.
    If data is uniform (or n < 2), returns empty tuple -> single bucket.
    """
    if max_k < 2:
        raise ValueError(f"max_k must be >= 2, got {max_k}")
    if not (0.0 <= gvf_threshold <= 1.0):
        raise ValueError(f"gvf_threshold must be in [0, 1], got {gvf_threshold}")
    n = len(char_lens)
    if n < 2 or char_lens[0] == char_lens[-1]:
        return ()

    sample = _sample_sorted(char_lens, JENKS_SAMPLE_SIZE)
    sampled = len(sample) < n
    if sampled:
        LOGGER.debug(
            "jenks: sampled %d of %d char_lens for clustering",
            len(sample),
            n,
        )

    best_breaks: list[int] = []
    for k in range(2, min(max_k, len(sample)) + 1):
        breaks = _jenks_breaks(sample, k)
        if not breaks:
            break
        gvf = _goodness_of_variance_fit(char_lens, breaks)
        best_breaks = breaks
        LOGGER.debug("jenks k=%d gvf=%.4f breaks=%s", k, gvf, breaks)
        if gvf >= gvf_threshold:
            break
    unique_breaks = sorted(set(best_breaks))
    return tuple(b for b in unique_breaks if b > 0)


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


def _binary_probe_max_safe(
    worker: EmbeddingWorker,
    sample_sentences: list[str],
    candidate_list: list[int],
    *,
    log_cuda_memory_warn: bool = False,
) -> tuple[int | None, list[tuple[int, float]]]:
    """Binary search for the largest candidate that succeeds without OOM.

    Returns (max_safe_candidate, attempted_list_with_timings).
    """
    n_sample = len(sample_sentences)
    eligible = [c for c in candidate_list if c <= n_sample]
    if not eligible:
        return None, []

    attempted: list[tuple[int, float]] = []
    lo, hi = 0, len(eligible) - 1
    best_safe_idx: int | None = None

    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = eligible[mid]
        if log_cuda_memory_warn:
            _maybe_warn_cuda_memory_before_probe(candidate)
        try:
            t0 = time.perf_counter()
            _ = worker.encode_batch(sample_sentences[:candidate])
            elapsed = time.perf_counter() - t0
            attempted.append((candidate, elapsed / candidate))
            best_safe_idx = mid
            lo = mid + 1
        except EmbeddingError as exc:
            LOGGER.warning(
                "binary probe failed at size=%d: %s",
                candidate,
                exc,
            )
            hi = mid - 1

    if best_safe_idx is None:
        return None, attempted
    return eligible[best_safe_idx], attempted


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

    candidate_list = sorted({c for c in candidates if c > 0})
    if not candidate_list:
        return fallback_batch_size

    max_safe, attempted = _binary_probe_max_safe(
        worker,
        sample_sentences,
        candidate_list,
        log_cuda_memory_warn=log_cuda_memory_warn,
    )

    if strategy is ProbeStrategy.MAX_SUCCESSFUL:
        if max_safe is None:
            LOGGER.info(
                "probe had no valid candidates, using fallback_batch_size=%d",
                fallback_batch_size,
            )
            return fallback_batch_size
        LOGGER.info(
            "probe complete: strategy=%s tested=%s winner=%d",
            strategy.value,
            ", ".join(f"{bs}:{lat:.6f}s_per_sent" for bs, lat in attempted),
            max_safe,
        )
        return max_safe

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
    char_len_bucketing: bool = False,
    char_len_bucket_edges: tuple[int, ...] | None = None,
    max_buckets: int = DEFAULT_MAX_BUCKETS,
    gvf_threshold: float = DEFAULT_GVF_THRESHOLD,
) -> PipelineResult:
    t0 = time.perf_counter()
    in_dir = Path(input_directory)
    out_path = Path(output_path)
    strategy = (
        probe_strategy
        if isinstance(probe_strategy, ProbeStrategy)
        else ProbeStrategy(probe_strategy)
    )
    use_jenks = char_len_bucketing and char_len_bucket_edges is None
    if char_len_bucketing and char_len_bucket_edges is not None:
        edges_eff: tuple[int, ...] | None = char_len_bucket_edges
        _validate_char_len_bucket_edges(edges_eff)
    elif not char_len_bucketing:
        edges_eff = None
    else:
        edges_eff = None
    resolved_probe = _resolved_probe_candidates(
        probe_batch_sizes,
        max_probe_batch,
        char_len_bucketing=char_len_bucketing,
    )
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
                probe_window = max(resolved_probe)
                min_sentences_to_probe = min(resolved_probe)

                def _process_metas_slice(
                    metas_slice: list[SentenceMeta],
                    *,
                    log_name: str,
                    sentence_count: int,
                ) -> None:
                    nonlocal total_batches
                    sentences = [m.text for m in metas_slice]
                    if len(sentences) >= min_sentences_to_probe:
                        head_ceiling = probe_batch_size(
                            worker,
                            sentences[:probe_window],
                            candidates=resolved_probe,
                            fallback_batch_size=batch_size,
                            strategy=strategy,
                            probe_epsilon=probe_epsilon,
                            log_cuda_memory_warn=probe_log_cuda_memory,
                        )
                        tail_ceiling = probe_batch_size(
                            worker,
                            sentences[-probe_window:],
                            candidates=resolved_probe,
                            fallback_batch_size=batch_size,
                            strategy=strategy,
                            probe_epsilon=probe_epsilon,
                            log_cuda_memory_warn=probe_log_cuda_memory,
                        )
                        selected = min(head_ceiling, tail_ceiling)
                        LOGGER.info(
                            "%s probe ceilings: head=%d tail=%d final=%d",
                            log_name,
                            head_ceiling,
                            tail_ceiling,
                            selected,
                        )
                    else:
                        selected = batch_size
                    char_budget_local = selected * CHAR_BUDGET_PER_SENTENCE
                    LOGGER.info(
                        "%s batch controls: sentences=%d batch_size=%d char_budget=%d",
                        log_name,
                        sentence_count,
                        selected,
                        char_budget_local,
                    )
                    for batch in batch_sentences_dynamic_cap(
                        metas_slice,
                        batch_size_ceiling=selected,
                        char_budget=char_budget_local,
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
                    if char_len_bucketing:
                        if use_jenks:
                            file_char_lens = [m.char_len for m in file_metas_sorted]
                            jenks_edges = _jenks_auto_bucket_edges(
                                file_char_lens,
                                max_k=max_buckets,
                                gvf_threshold=gvf_threshold,
                            )
                            if jenks_edges:
                                _validate_char_len_bucket_edges(jenks_edges)
                                active_edges = jenks_edges
                            else:
                                active_edges = None
                            LOGGER.info(
                                "file=%s jenks edges=%s (from %d char_lens)",
                                path.name,
                                active_edges,
                                len(file_char_lens),
                            )
                        else:
                            active_edges = edges_eff
                        if active_edges:
                            buckets = _split_sorted_metas_by_char_len_edges(
                                file_metas_sorted, active_edges
                            )
                        else:
                            buckets = [file_metas_sorted]
                        for bi, bucket_metas in enumerate(buckets):
                            if active_edges:
                                band = _char_len_band_label(
                                    bucket_metas[0].char_len, active_edges
                                )
                            else:
                                band = "[all]"
                            _process_metas_slice(
                                bucket_metas,
                                log_name=(
                                    f"file={path.name} bucket_index={bi} band={band}"
                                ),
                                sentence_count=len(bucket_metas),
                            )
                    else:
                        _process_metas_slice(
                            file_metas_sorted,
                            log_name=f"file={path.name}",
                            sentence_count=len(file_metas_sorted),
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
