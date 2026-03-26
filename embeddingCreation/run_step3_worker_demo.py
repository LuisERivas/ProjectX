#!/usr/bin/env python3
"""
Step 3 demo runner: initialize worker, encode a batch, then shutdown.

Usage:
  python3 run_step3_worker_demo.py
  python3 run_step3_worker_demo.py --sentences "hello" "world"
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from embedding_worker import EmbeddingWorker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sentences",
        nargs="+",
        default=[
            "Step 3 demo sentence one.",
            "Step 3 demo sentence two.",
        ],
        help="one or more sentences to encode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    worker = EmbeddingWorker()

    t0 = time.perf_counter()
    try:
        worker.init()
        print("init: ok")
        print("stats_after_init:", worker.stats)

        out = worker.encode_batch(args.sentences)
        elapsed = time.perf_counter() - t0

        print("encode: ok")
        print("output_shape:", out.shape)
        print("output_dtype:", out.dtype)
        if out.shape[0] != len(args.sentences):
            print("error: output count mismatch", file=sys.stderr)
            return 1
        if out.shape[1] != 2048:
            print("error: output dim mismatch (expected 2048)", file=sys.stderr)
            return 1

        norms = np.linalg.norm(out.astype(np.float32), axis=1)
        print("norm_min:", float(norms.min()))
        print("norm_max:", float(norms.max()))
        print("stats_after_encode:", worker.stats)
        print("elapsed_s:", round(elapsed, 4))
        return 0
    except Exception as exc:
        print(f"demo_failed: {exc!r}", file=sys.stderr)
        return 1
    finally:
        try:
            worker.shutdown()
            print("shutdown: ok")
        except Exception as exc:
            print(f"shutdown_failed: {exc!r}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

