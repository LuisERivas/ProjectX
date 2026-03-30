#!/usr/bin/env python3
"""
Step 12 benchmark runner for Jetson performance tuning.

Sweeps batch sizes, captures throughput/memory/correctness, writes a JSON report,
and prints a human-readable comparison table.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from batch_builder import batch_sentences
from binary_reader import verify_file
from binary_writer import EmbeddingWriter
from embedding_worker import EmbeddingError, EmbeddingWorker, MODEL_ID
from file_reader import read_text_files

LOGGER = logging.getLogger("benchmark_jetson")
EXPECTED_DIM = 2048
MAX_PENDING_WRITES = 2


@dataclass(frozen=True)
class BenchmarkResult:
    batch_size: int
    dtype_used: str
    device: str
    total_sentences: int
    total_batches: int
    model_load_time_s: float
    total_encode_time_s: float
    total_pipeline_time_s: float
    sentences_per_sec: float
    avg_batch_latency_s: float
    peak_cuda_memory_bytes: int
    peak_process_rss_bytes: int
    oom_occurred: bool
    verification_ok: bool
    records_written: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=100,
        help="number of synthetic sentences (ignored if --corpus-dir is set)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=None,
        help="directory with .txt files to use instead of synthetic corpus",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="benchmark_report.json",
        help="path to write JSON benchmark report",
    )
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument(
        "--soak",
        action="store_true",
        help="run soak test at recommended batch size",
    )
    parser.add_argument("--soak-sentences", type=int, default=500)
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="keep generated benchmark output files",
    )
    return parser.parse_args()


def _check_requirements() -> tuple[bool, str | None]:
    try:
        import icu  # noqa: F401
    except Exception as exc:
        return False, f"PyICU unavailable: {exc}"

    try:
        import torch
    except Exception as exc:
        return False, f"PyTorch unavailable: {exc}"

    if not torch.cuda.is_available():
        return False, "CUDA unavailable (torch.cuda.is_available() == False)"
    return True, None


def _generate_corpus(n: int) -> list[str]:
    if n <= 0:
        raise ValueError(f"corpus size must be > 0, got {n}")
    return [
        (
            f"Benchmark sentence number {i}. "
            "The quick brown fox jumps over the lazy dog near the river bank."
        )
        for i in range(n)
    ]


def _load_corpus_from_dir(corpus_dir: str | Path) -> tuple[list[str], int]:
    from sentence_splitter import split_sentences

    root = Path(corpus_dir)
    rows = list(read_text_files(root))
    sentences: list[str] = []
    for _, text in rows:
        sentences.extend(split_sentences(text, locale="en_US"))
    return sentences, len(rows)


def _read_rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return 0
    return 0


def _read_memtotal_bytes() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return 0
    return 0


def _read_jetpack_version() -> str:
    p = Path("/etc/nv_tegra_release")
    if not p.exists():
        return "unknown"
    try:
        text = p.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return "unknown"
    return text or "unknown"


def _read_nvpmodel() -> str:
    try:
        proc = subprocess.run(
            ["nvpmodel", "-q"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        out = (proc.stdout or "").strip()
        if proc.returncode == 0 and out:
            return out
    except Exception:
        pass
    return "unavailable"


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cuda error" in msg) or ("oom" in msg)


def run_batch_size_benchmark(
    sentences: list[str],
    batch_size: int,
    worker: EmbeddingWorker,
    output_dir: Path,
    warmup_batches: int = 1,
) -> BenchmarkResult:
    import torch

    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    total_candidate_batches = (len(sentences) + batch_size - 1) // batch_size
    warmup_count = min(max(0, warmup_batches), total_candidate_batches)

    LOGGER.info(
        "starting batch-size run: batch_size=%d total_batches=%d warmup_batches=%d",
        batch_size,
        total_candidate_batches,
        warmup_count,
    )

    if batch_size > 8:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            if free_bytes < (1 * 1024 * 1024 * 1024):
                LOGGER.warning(
                    "low free CUDA memory before batch_size=%d run: free=%.2f GB total=%.2f GB",
                    batch_size,
                    free_bytes / (1024**3),
                    total_bytes / (1024**3),
                )
        except Exception:
            pass

    if warmup_count > 0:
        for idx, wb in enumerate(batch_sentences(sentences, batch_size=batch_size, start_id=0), 1):
            _ = worker.encode_batch(wb.sentences)
            if idx >= warmup_count:
                break

    output_path = output_dir / f"benchmark_bs{batch_size}.bin"
    if output_path.exists():
        output_path.unlink()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    peak_rss = _read_rss_bytes()
    run_start = time.perf_counter()
    records_written = 0
    verification_ok = False
    oom_occurred = False

    stats_before = worker.stats
    per_batch_latencies: list[float] = []
    try:
        with EmbeddingWriter(output_path) as writer:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending_writes: deque[Future[None]] = deque()
                for batch in batch_sentences(sentences, batch_size=batch_size, start_id=0):
                    peak_rss = max(peak_rss, _read_rss_bytes())
                    t0 = time.perf_counter()
                    embeddings = worker.encode_batch(batch.sentences)
                    per_batch_latencies.append(time.perf_counter() - t0)
                    pending_writes.append(pool.submit(writer.write_batch, batch.ids, embeddings))
                    if len(pending_writes) >= MAX_PENDING_WRITES:
                        pending_writes.popleft().result()
                    peak_rss = max(peak_rss, _read_rss_bytes())
                while pending_writes:
                    pending_writes.popleft().result()
            records_written = writer.records_written
    except (EmbeddingError, RuntimeError) as exc:
        if _is_oom_error(exc):
            oom_occurred = True
            LOGGER.warning(
                "OOM occurred during batch_size=%d run: %s",
                batch_size,
                exc,
            )
        else:
            raise

    stats_after = worker.stats
    total_pipeline_time_s = time.perf_counter() - run_start
    total_encode_time_s = float(stats_after["total_encode_time_s"]) - float(
        stats_before["total_encode_time_s"]
    )
    total_batches = int(stats_after["total_batches"]) - int(stats_before["total_batches"])
    total_sentences = int(stats_after["total_sentences"]) - int(stats_before["total_sentences"])
    model_load_time_s = float(stats_after["init_time_s"] or 0.0)

    try:
        peak_cuda_memory_bytes = int(torch.cuda.max_memory_allocated())
    except Exception:
        peak_cuda_memory_bytes = 0

    if not oom_occurred:
        report = verify_file(output_path)
        verification_ok = report.ok
        if not report.ok:
            LOGGER.error("verification failed for batch_size=%d: %s", batch_size, report.errors)

    sentences_per_sec = (
        (total_sentences / total_encode_time_s)
        if total_encode_time_s > 0 and total_sentences > 0
        else 0.0
    )
    avg_batch_latency_s = (
        (total_encode_time_s / total_batches) if total_batches > 0 else 0.0
    )

    if len(per_batch_latencies) >= 6:
        third = max(1, len(per_batch_latencies) // 3)
        early_avg = sum(per_batch_latencies[:third]) / third
        late_avg = sum(per_batch_latencies[-third:]) / third
        if late_avg > early_avg * 1.15:
            LOGGER.warning(
                "possible thermal throttling at batch_size=%d: early_avg=%.4f late_avg=%.4f",
                batch_size,
                early_avg,
                late_avg,
            )

    return BenchmarkResult(
        batch_size=batch_size,
        dtype_used=str(stats_after.get("dtype") or "unknown"),
        device=str(stats_after.get("device") or "unknown"),
        total_sentences=total_sentences,
        total_batches=total_batches,
        model_load_time_s=model_load_time_s,
        total_encode_time_s=total_encode_time_s,
        total_pipeline_time_s=total_pipeline_time_s,
        sentences_per_sec=sentences_per_sec,
        avg_batch_latency_s=avg_batch_latency_s,
        peak_cuda_memory_bytes=peak_cuda_memory_bytes,
        peak_process_rss_bytes=peak_rss,
        oom_occurred=oom_occurred,
        verification_ok=verification_ok,
        records_written=records_written,
    )


def _recommend_batch_size(results: list[BenchmarkResult]) -> tuple[int | None, str]:
    passing = [r for r in results if (not r.oom_occurred) and r.verification_ok]
    if not passing:
        return None, "no passing configuration (OOM and/or verification failures)"

    max_sps = max(r.sentences_per_sec for r in passing)
    near_top = [r for r in passing if r.sentences_per_sec >= (max_sps * 0.95)]
    chosen = min(near_top, key=lambda r: r.batch_size)
    reason = (
        "selected smallest batch size within 5% of max throughput "
        f"(max={max_sps:.2f} sent/s)"
    )
    return chosen.batch_size, reason


def _print_comparison_table(results: list[BenchmarkResult]) -> None:
    print(
        "batch_size | dtype    | sent/sec | avg_batch_ms | peak_cuda_MB | peak_rss_MB | oom | verify"
    )
    print("-----------+----------+----------+--------------+--------------+-------------+-----+-------")
    for r in sorted(results, key=lambda x: x.batch_size):
        print(
            f"{r.batch_size:<10} | "
            f"{r.dtype_used:<8} | "
            f"{r.sentences_per_sec:>8.2f} | "
            f"{(r.avg_batch_latency_s * 1000.0):>12.2f} | "
            f"{(r.peak_cuda_memory_bytes / (1024**2)):>12.2f} | "
            f"{(r.peak_process_rss_bytes / (1024**2)):>11.2f} | "
            f"{'yes' if r.oom_occurred else 'no ':<3} | "
            f"{'ok' if r.verification_ok else 'bad'}"
        )
    print("Note: Jetson uses unified memory; CUDA and RSS numbers overlap.")


def _run_soak_test(
    worker: EmbeddingWorker,
    *,
    batch_size: int,
    soak_sentences: int,
    output_dir: Path,
) -> dict[str, Any]:
    import torch

    LOGGER.info(
        "starting soak test: batch_size=%d soak_sentences=%d",
        batch_size,
        soak_sentences,
    )
    sentences = _generate_corpus(soak_sentences)
    output_path = output_dir / f"soak_bs{batch_size}.bin"
    if output_path.exists():
        output_path.unlink()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    timeline: list[dict[str, float | int]] = []

    with EmbeddingWriter(output_path) as writer:
        for idx, batch in enumerate(batch_sentences(sentences, batch_size=batch_size, start_id=0), 1):
            embeddings = worker.encode_batch(batch.sentences)
            writer.write_batch(batch.ids, embeddings)
            if idx % 10 == 0:
                cuda_alloc = int(torch.cuda.memory_allocated())
                rss = _read_rss_bytes()
                timeline.append(
                    {
                        "batch_index": idx,
                        "cuda_allocated_mb": round(cuda_alloc / (1024**2), 2),
                        "rss_mb": round(rss / (1024**2), 2),
                    }
                )

    report = verify_file(output_path)
    leak_warning = False
    if len(timeline) >= 5:
        last_five = [float(t["cuda_allocated_mb"]) for t in timeline[-5:]]
        monotonic = all(last_five[i] < last_five[i + 1] for i in range(len(last_five) - 1))
        peak = max(last_five) if last_five else 0.0
        delta = last_five[-1] - last_five[0] if len(last_five) >= 2 else 0.0
        if monotonic and peak > 0 and delta > (0.10 * peak):
            leak_warning = True

    return {
        "batch_size": batch_size,
        "soak_sentences": soak_sentences,
        "verification_ok": report.ok,
        "memory_timeline": timeline,
        "leak_warning": leak_warning,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    ok, reason = _check_requirements()
    if not ok:
        raise RuntimeError(f"requirements check failed: {reason}")

    output_dir = Path(mkdtemp(prefix="step12_benchmark_"))
    corpus_source = "synthetic"
    file_count = 0
    if args.corpus_dir:
        sentences, file_count = _load_corpus_from_dir(args.corpus_dir)
        corpus_source = "user"
    else:
        sentences = _generate_corpus(args.corpus_size)
    if not sentences:
        raise RuntimeError("corpus is empty; nothing to benchmark")

    LOGGER.info(
        "benchmark corpus ready: source=%s file_count=%d sentence_count=%d",
        corpus_source,
        file_count,
        len(sentences),
    )

    baseline_rss = _read_rss_bytes()
    device_name = torch.cuda.get_device_name(0)
    jetpack = _read_jetpack_version()
    nvpmodel = _read_nvpmodel()

    batch_sizes = sorted(set(args.batch_sizes))
    if any(bs <= 0 for bs in batch_sizes):
        raise ValueError(f"batch sizes must be > 0, got {batch_sizes}")

    worker = EmbeddingWorker()
    results: list[BenchmarkResult] = []
    soak_result: dict[str, Any] | None = None
    try:
        worker.init()
        post_init_rss = _read_rss_bytes()
        LOGGER.info(
            "worker initialized: device=%s dtype=%s init_time_s=%s",
            worker.stats.get("device"),
            worker.stats.get("dtype"),
            worker.stats.get("init_time_s"),
        )
        for bs in batch_sizes:
            result = run_batch_size_benchmark(
                sentences=sentences,
                batch_size=bs,
                worker=worker,
                output_dir=output_dir,
                warmup_batches=args.warmup_batches,
            )
            results.append(result)
            if result.oom_occurred:
                LOGGER.warning("OOM at batch size %d; skipping larger batch sizes", bs)
                break

        recommended_batch_size, recommendation_reason = _recommend_batch_size(results)
        if args.soak and recommended_batch_size is not None:
            soak_result = _run_soak_test(
                worker,
                batch_size=recommended_batch_size,
                soak_sentences=args.soak_sentences,
                output_dir=output_dir,
            )
    finally:
        worker.shutdown()

    post_shutdown_rss = _read_rss_bytes()

    report: dict[str, Any] = {
        "hardware": {
            "device_name": device_name,
            "jetpack_version": jetpack,
            "total_memory_bytes": _read_memtotal_bytes(),
            "nvpmodel": nvpmodel,
        },
        "model": {
            "model_id": MODEL_ID,
            "truncate_dim": EXPECTED_DIM,
            "attn_implementation": "sdpa",
            "dtype_used": str(results[0].dtype_used if results else "unknown"),
        },
        "corpus": {
            "sentence_count": len(sentences),
            "file_count": file_count,
            "source": corpus_source,
        },
        "memory_baseline": {
            "rss_before_model_load_bytes": baseline_rss,
            "rss_after_model_load_bytes": post_init_rss,
            "rss_after_shutdown_bytes": post_shutdown_rss,
        },
        "warmup_batches": args.warmup_batches,
        "results": [asdict(r) for r in results],
        "recommendation": {
            "batch_size": recommended_batch_size,
            "reason": recommendation_reason,
        },
        "soak": soak_result,
    }

    _print_comparison_table(results)
    if report["recommendation"]["batch_size"] is None:
        print(f"recommendation: none ({report['recommendation']['reason']})")
    else:
        best = report["recommendation"]["batch_size"]
        best_row = next(r for r in results if r.batch_size == best)
        print(
            f"recommendation: batch_size={best} "
            f"(stable throughput {best_row.sentences_per_sec:.2f} sent/s; {report['recommendation']['reason']})"
        )
    if soak_result is not None:
        print(f"soak: {'warning' if soak_result['leak_warning'] else 'ok'}")

    report_path = Path(args.output_report)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report_written: {report_path.resolve()}")

    if args.keep_outputs:
        print(f"outputs_kept: {output_dir}")
    else:
        shutil.rmtree(output_dir, ignore_errors=True)
    return report


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    try:
        _ = run_benchmark(args)
        return 0
    except Exception as exc:
        print(f"benchmark_failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
