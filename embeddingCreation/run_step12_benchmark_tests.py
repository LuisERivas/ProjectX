#!/usr/bin/env python3
"""
Step 12 test runner: executes benchmark sweep + soak test in one run.

This wraps benchmark_jetson.py so you do not have to run multiple commands.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from benchmark_jetson import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 8, 16],
        help="batch sizes to test",
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=100,
        help="synthetic corpus size when --corpus-dir is not provided",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=None,
        help="optional directory of .txt files to use as corpus",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="number of warmup batches",
    )
    parser.add_argument(
        "--soak-sentences",
        type=int,
        default=500,
        help="soak-test sentence count",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=".",
        help="directory where JSON reports will be written",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="keep generated benchmark binary files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    benchmark_report = reports_dir / f"step12_benchmark_{ts}.json"
    print("step12_runner: starting benchmark sweep (+ soak) in one pass")
    report = run_benchmark(
        argparse.Namespace(
            batch_sizes=args.batch_sizes,
            corpus_size=args.corpus_size,
            corpus_dir=args.corpus_dir,
            output_report=str(benchmark_report),
            warmup_batches=args.warmup_batches,
            soak=True,
            soak_sentences=args.soak_sentences,
            keep_outputs=args.keep_outputs,
        )
    )

    recommendation = report.get("recommendation", {})
    rec_bs = recommendation.get("batch_size")
    rec_reason = recommendation.get("reason")
    print(f"step12_runner: recommendation batch_size={rec_bs} reason={rec_reason}")

    soak = report.get("soak") or {}
    print(f"step12_runner: soak verification_ok={soak.get('verification_ok')}")
    print(f"step12_runner: soak leak_warning={soak.get('leak_warning')}")
    print(f"step12_runner: benchmark_report={benchmark_report.resolve()}")
    print("step12_runner: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
