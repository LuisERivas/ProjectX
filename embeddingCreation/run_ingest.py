#!/usr/bin/env python3
"""
Minimal CLI runner for the embedding ingestion pipeline.

Usage:
  python3 run_ingest.py --input /path/to/folder --output /path/to/output.bin
  python3 run_ingest.py --input ... --output ... --batch-size 32
"""

from __future__ import annotations

import argparse
import logging
import sys

from batch_builder import DEFAULT_BATCH_SIZE
from ingest_pipeline import (
    DEFAULT_CHAR_LEN_BUCKET_EDGES,
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
        "--probe-strategy",
        type=str,
        choices=[s.value for s in ProbeStrategy],
        default=ProbeStrategy.MIN_LATENCY_PER_SENT.value,
        help=(
            "how to pick the batch ceiling after probe encodes: min_latency_per_sent (default), "
            "max_successful (largest batch that succeeded), or epsilon (largest batch within "
            "(1+eps) of best latency per sentence)"
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
            "drops candidates above N"
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
            "split each file into char_len bands (default edges "
            f"{','.join(str(e) for e in DEFAULT_CHAR_LEN_BUCKET_EDGES)}), probe and batch each "
            "band separately; default probe ladder cap becomes 16384 (many probe encodes; use "
            "--max-probe-batch or --probe-batch-sizes to limit)"
        ),
    )
    parser.add_argument(
        "--bucket-edges",
        type=_parse_bucket_edges,
        default=None,
        metavar="LIST",
        help=(
            'char_len bucket upper bounds, comma-separated (strictly increasing), e.g. '
            '"16,32,64,128,256,512,1024"; requires --char-len-buckets; default is built-in edges'
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        print("run_ingest: --batch-size must be >= 1", file=sys.stderr)
        return 2
    if args.probe_epsilon < 0:
        print("run_ingest: --probe-epsilon must be >= 0", file=sys.stderr)
        return 2
    if args.max_probe_batch is not None and args.max_probe_batch < 1:
        print("run_ingest: --max-probe-batch must be >= 1 when set", file=sys.stderr)
        return 2
    if args.bucket_edges is not None and not args.char_len_buckets:
        print(
            "run_ingest: --bucket-edges requires --char-len-buckets",
            file=sys.stderr,
        )
        return 2
    logging.basicConfig(level=logging.INFO)

    result = run_pipeline(
        args.input,
        args.output,
        batch_size=args.batch_size,
        probe_batch_sizes=args.probe_batch_sizes,
        max_probe_batch=args.max_probe_batch,
        probe_strategy=args.probe_strategy,
        probe_epsilon=args.probe_epsilon,
        probe_log_cuda_memory=args.probe_log_cuda_memory,
        char_len_bucketing=args.char_len_buckets,
        char_len_bucket_edges=args.bucket_edges,
    )

    print("pipeline_result:")
    print(f"- batch_size: {args.batch_size}")
    print(f"- probe_strategy: {args.probe_strategy}")
    print(f"- probe_batch_sizes: {args.probe_batch_sizes}")
    print(f"- max_probe_batch: {args.max_probe_batch}")
    print(f"- char_len_bucketing: {args.char_len_buckets}")
    print(f"- bucket_edges: {args.bucket_edges}")
    print(f"- input_directory: {result.input_directory}")
    print(f"- output_path: {result.output_path}")
    print(f"- files_discovered: {result.files_discovered}")
    print(f"- files_read: {result.files_read}")
    print(f"- files_skipped: {result.files_skipped}")
    print(f"- total_sentences: {result.total_sentences}")
    print(f"- total_batches: {result.total_batches}")
    print(f"- records_written: {result.records_written}")
    print(f"- elapsed_seconds: {result.elapsed_seconds:.4f}")
    print(f"- success: {result.success}")
    print(f"- errors: {result.errors}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
