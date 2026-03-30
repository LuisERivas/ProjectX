#!/usr/bin/env python3
"""
Step 11 smoke runner: full ingestion pipeline with real model.

Usage:
  python3 run_step11_pipeline_smoke.py
  python3 run_step11_pipeline_smoke.py --keep-fixtures
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from binary_reader import verify_file
from ingest_pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        default="tmp_step11_smoke",
        help="temporary fixture directory name (default: tmp_step11_smoke)",
    )
    parser.add_argument(
        "--keep-fixtures",
        action="store_true",
        help="keep generated fixture files after run",
    )
    return parser.parse_args()


def _create_fixtures(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.txt").write_text(
        "First sentence. Second sentence? Third sentence!\n",
        encoding="utf-8",
    )
    (root / "b.txt").write_text(
        "Dr. Smith arrived. He said hello. This is a pipeline smoke test.",
        encoding="utf-8",
    )
    (root / "c.txt").write_text("Final fixture file for Step 11.", encoding="utf-8")


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


def main() -> int:
    args = parse_args()
    root = Path(args.workspace)
    output_path = root / "step11_output.bin"

    ok, reason = _check_requirements()
    if not ok:
        print("smoke_failed: missing runtime requirements", file=sys.stderr)
        print(f"details: {reason}", file=sys.stderr)
        return 2

    try:
        _create_fixtures(root)
        print(f"fixtures_dir: {root.resolve()}")

        result = run_pipeline(root, output_path, batch_size=8, locale="en_US")
        print("pipeline_result:")
        print(f"- input_directory: {result.input_directory}")
        print(f"- output_path: {result.output_path}")
        print(f"- files_discovered: {result.files_discovered}")
        print(f"- files_read: {result.files_read}")
        print(f"- total_sentences: {result.total_sentences}")
        print(f"- total_batches: {result.total_batches}")
        print(f"- records_written: {result.records_written}")
        print(f"- elapsed_seconds: {result.elapsed_seconds:.4f}")
        print(f"- success: {result.success}")
        print(f"- errors: {result.errors}")

        report = verify_file(output_path)
        print("verification_report:")
        print(f"- record_count: {report.record_count}")
        print(f"- id_min: {report.id_min}")
        print(f"- id_max: {report.id_max}")
        print(f"- all_finite: {report.all_finite}")
        print(f"- norms_min: {report.norms_min}")
        print(f"- norms_max: {report.norms_max}")
        print(f"- norms_mean: {report.norms_mean}")
        print(f"- ok: {report.ok}")
        print(f"- errors: {report.errors}")
        print("smoke: ok")
        return 0
    except Exception as exc:
        print(f"smoke_failed: {exc!r}", file=sys.stderr)
        return 1
    finally:
        if args.keep_fixtures:
            print("cleanup: skipped (--keep-fixtures)")
        else:
            shutil.rmtree(root, ignore_errors=True)
            print("cleanup: ok")


if __name__ == "__main__":
    raise SystemExit(main())
