#!/usr/bin/env python3
"""
Run the same embedding ingest pipeline as run_ingest.py and write a timing report.

Each timed step includes its share of total pipeline wall time (percent). Overlap note:
the split producer thread runs concurrently with encode/write on the main thread, and
encode vs write overlap when pending writes are pipelined, so listed percents may sum
to more than 100%.

Usage:
  py -3 run_ingest_latency.py --input /path/to/folder --output /path/to/output.bin
  py -3 run_ingest_latency.py ... --latency-file latencyData.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from batch_builder import DEFAULT_BATCH_SIZE
from ingest_pipeline import (
    DEFAULT_CHAR_LEN_BUCKET_EDGES,
    DEFAULT_GVF_THRESHOLD,
    DEFAULT_MAX_BUCKETS,
    DEFAULT_NON_BUCKET_PROBE_CAP,
    DEFAULT_PROBE_EPSILON,
    ProbeStrategy,
    run_pipeline,
)


def _parse_probe_batch_sizes(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("comma-separated list is empty")
    try:
        return tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer in probe batch list: {exc}") from exc


def _parse_bucket_edges(value: str) -> tuple[int, ...]:
    return _parse_probe_batch_sizes(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="path to input folder containing text files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="path to output .bin file",
    )
    parser.add_argument(
        "--latency-file",
        type=Path,
        default=Path("latencyData.txt"),
        metavar="PATH",
        help="where to write the latency report (default: latencyData.txt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help=(
            "fallback batch size when a file has fewer sentences than the minimum probe "
            f"candidate (default: {DEFAULT_BATCH_SIZE}); also used if all probe candidates fail"
        ),
    )
    parser.add_argument(
        "--probe-ladder",
        type=str,
        choices=("full", "short"),
        default="full",
        help=(
            "probe candidate ladder to use when --probe-batch-sizes is not provided: "
            "full uses 64..16384 powers-of-two; short uses 64..1024. Ignored when "
            "--probe-batch-sizes is provided"
        ),
    )
    parser.add_argument(
        "--probe-single-window",
        action="store_true",
        help=(
            "use only a head-window probe in non-bucketing mode (faster startup, less conservative); "
            "default probes head and tail and takes min"
        ),
    )
    parser.add_argument(
        "--probe-strategy",
        type=str,
        choices=[s.value for s in ProbeStrategy],
        default=ProbeStrategy.MIN_LATENCY_PER_SENT.value,
        help=(
            "how to pick the batch ceiling after probe encodes: min_latency_per_sent (default) "
            "targets best probe-time latency per sentence; max_successful picks the largest passing "
            "batch (often best throughput); epsilon picks the largest batch within (1+eps) of best "
            "latency per sentence"
        ),
    )
    parser.add_argument(
        "--probe-epsilon",
        type=float,
        default=DEFAULT_PROBE_EPSILON,
        metavar="X",
        help=(
            "for --probe-strategy epsilon: tie band as fraction of best latency/sentence "
            f"(default: {DEFAULT_PROBE_EPSILON})"
        ),
    )
    parser.add_argument(
        "--max-probe-batch",
        type=int,
        default=None,
        metavar="N",
        help=(
            "cap probe ladder: when set without --probe-batch-sizes, try batch sizes up to N "
            "from the built-in ladder (64..16384 powers-of-two steps). When set with --probe-batch-sizes, "
            "drops candidates above N. When omitted, non-bucketing mode defaults to "
            f"{DEFAULT_NON_BUCKET_PROBE_CAP} and bucketing mode defaults to 16384"
        ),
    )
    parser.add_argument(
        "--probe-batch-sizes",
        type=_parse_probe_batch_sizes,
        default=None,
        metavar="LIST",
        help='explicit probe sizes, e.g. "64,128,256,512" (comma-separated)',
    )
    parser.add_argument(
        "--probe-log-cuda-memory",
        action="store_true",
        help="log a CUDA free-memory warning before each probe encode when free < 1 GiB",
    )
    parser.add_argument(
        "--char-len-buckets",
        action="store_true",
        help=(
            "split each file into char_len bands, probe and batch each band separately. When "
            "--bucket-edges is omitted, Jenks natural breaks choose per-file edges; when "
            "--bucket-edges is provided, fixed edges are used. Probe ceilings are cached by "
            "approximate band signature so similar bands across files can skip repeat probes. "
            "Default probe ladder cap becomes 16384 (use --max-probe-batch or "
            "--probe-batch-sizes to limit)"
        ),
    )
    parser.add_argument(
        "--bucket-edges",
        type=_parse_bucket_edges,
        default=None,
        metavar="LIST",
        help=(
            'char_len bucket upper bounds, comma-separated (strictly increasing), e.g. '
            f'"{",".join(str(e) for e in DEFAULT_CHAR_LEN_BUCKET_EDGES)}"; requires '
            "--char-len-buckets. When set, skips per-file Jenks/GVF edge discovery and uses "
            "these fixed edges"
        ),
    )
    parser.add_argument(
        "--max-buckets",
        type=int,
        default=DEFAULT_MAX_BUCKETS,
        metavar="K",
        help=(
            f"maximum number of Jenks char_len buckets per file (default: {DEFAULT_MAX_BUCKETS}); "
            "only used when --char-len-buckets is set without --bucket-edges"
        ),
    )
    parser.add_argument(
        "--gvf-threshold",
        type=float,
        default=DEFAULT_GVF_THRESHOLD,
        metavar="X",
        help=(
            "Jenks goodness-of-variance-fit target: stop increasing k when GVF reaches this "
            f"level (default: {DEFAULT_GVF_THRESHOLD}); only used with --char-len-buckets "
            "without --bucket-edges"
        ),
    )
    return parser.parse_args()


def _write_latency_report(
    path: Path,
    *,
    events: list[tuple[str, float]],
    total_wall_s: float,
    pipeline_elapsed_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Embedding ingest latency report",
        "# total_wall_s is perf_counter() around run_pipeline() only (CLI parse + this write excluded).",
        "# pipeline_elapsed_s is the same span recorded inside run_pipeline (should match total_wall_s).",
        "# pct_of_total uses total_wall_s as the denominator for every row.",
        "# Threaded split + pipelined encode/write can make percents sum to more than 100%.",
        f"# total_wall_s={total_wall_s:.9f}",
        f"# pipeline_elapsed_s={pipeline_elapsed_s:.9f}",
        "#",
    ]
    if total_wall_s <= 0.0:
        denom = 1.0
    else:
        denom = total_wall_s
    for label, dur in events:
        pct = 100.0 * dur / denom
        lines.append(f"{label}\t{dur:.9f}\t{pct:.4f}%")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        print("run_ingest_latency: --batch-size must be >= 1", file=sys.stderr)
        return 2
    if args.probe_epsilon < 0:
        print("run_ingest_latency: --probe-epsilon must be >= 0", file=sys.stderr)
        return 2
    if args.max_probe_batch is not None and args.max_probe_batch < 1:
        print("run_ingest_latency: --max-probe-batch must be >= 1 when set", file=sys.stderr)
        return 2
    if args.bucket_edges is not None and not args.char_len_buckets:
        print(
            "run_ingest_latency: --bucket-edges requires --char-len-buckets",
            file=sys.stderr,
        )
        return 2
    if args.max_buckets < 2:
        print("run_ingest_latency: --max-buckets must be >= 2", file=sys.stderr)
        return 2
    if not (0.0 <= args.gvf_threshold <= 1.0):
        print("run_ingest_latency: --gvf-threshold must be in [0, 1]", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO)

    latency_events: list[tuple[str, float]] = []
    t_run = time.perf_counter()
    result = run_pipeline(
        args.input,
        args.output,
        batch_size=args.batch_size,
        probe_batch_sizes=args.probe_batch_sizes,
        max_probe_batch=args.max_probe_batch,
        probe_ladder=args.probe_ladder,
        probe_dual_window=not args.probe_single_window,
        probe_strategy=args.probe_strategy,
        probe_epsilon=args.probe_epsilon,
        probe_log_cuda_memory=args.probe_log_cuda_memory,
        char_len_bucketing=args.char_len_buckets,
        char_len_bucket_edges=args.bucket_edges,
        max_buckets=args.max_buckets,
        gvf_threshold=args.gvf_threshold,
        latency_events=latency_events,
    )
    total_wall_s = time.perf_counter() - t_run

    _write_latency_report(
        args.latency_file,
        events=latency_events,
        total_wall_s=total_wall_s,
        pipeline_elapsed_s=result.elapsed_seconds,
    )

    print("pipeline_result:")
    print(f"- batch_size: {args.batch_size}")
    print(f"- probe_strategy: {args.probe_strategy}")
    print(f"- probe_batch_sizes: {args.probe_batch_sizes}")
    print(f"- max_probe_batch: {args.max_probe_batch}")
    print(f"- probe_ladder: {args.probe_ladder}")
    print(f"- probe_single_window: {args.probe_single_window}")
    print(f"- char_len_bucketing: {args.char_len_buckets}")
    print(f"- bucket_edges: {args.bucket_edges}")
    print(f"- max_buckets: {args.max_buckets}")
    print(f"- gvf_threshold: {args.gvf_threshold}")
    print(f"- input_directory: {result.input_directory}")
    print(f"- output_path: {result.output_path}")
    print(f"- files_discovered: {result.files_discovered}")
    print(f"- files_read: {result.files_read}")
    print(f"- files_skipped: {result.files_skipped}")
    print(f"- total_sentences: {result.total_sentences}")
    print(f"- total_batches: {result.total_batches}")
    print(f"- records_written: {result.records_written}")
    print(f"- elapsed_seconds: {result.elapsed_seconds:.4f}")
    print(f"- total_wall_around_run_pipeline_s: {total_wall_s:.4f}")
    print(f"- success: {result.success}")
    print(f"- errors: {result.errors}")
    print(f"- latency_report: {args.latency_file.resolve()}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
