#!/usr/bin/env python3
"""
Minimal CLI runner for the embedding ingestion pipeline.

Usage:
  python3 run_ingest.py --input /path/to/folder --output /path/to/output.bin
"""

from __future__ import annotations

import argparse
import logging
import sys

from ingest_pipeline import run_pipeline


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    result = run_pipeline(args.input, args.output)

    print("pipeline_result:")
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
