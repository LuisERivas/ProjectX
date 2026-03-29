#!/usr/bin/env python3
"""
Step 5 smoke runner: file_reader -> sentence_splitter integration.

Usage:
  python3 run_step5_file_reader_smoke.py
  python3 run_step5_file_reader_smoke.py --keep-fixtures
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from file_reader import read_text_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        default="tmp_step5_smoke",
        help="temporary fixture directory name (default: tmp_step5_smoke)",
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
        "First sentence. Second sentence?\n\nThird sentence!\r\n",
        encoding="utf-8",
    )
    (root / "b.txt").write_text("Dr. Smith arrived. He said hello.", encoding="utf-8")
    (root / "ignore.bin").write_bytes(b"\x00\x01\x02")
    (root / "empty.txt").write_bytes(b"")


def main() -> int:
    args = parse_args()
    root = Path(args.workspace)

    try:
        from sentence_splitter import split_sentences
    except ModuleNotFoundError as exc:
        print(
            "smoke_failed: sentence_splitter dependency missing "
            "(likely PyICU). Install and retry.",
            file=sys.stderr,
        )
        print(f"details: {exc}", file=sys.stderr)
        return 2

    try:
        _create_fixtures(root)
        print(f"fixtures_dir: {root.resolve()}")

        rows = list(read_text_files(root))
        print(f"files_read: {len(rows)}")
        print(f"files_read_names: {[p.name for p, _ in rows]}")

        total_sentences = 0
        for path, text in rows:
            sentences = split_sentences(text)
            total_sentences += len(sentences)
            print(f"{path.name}: sentence_count={len(sentences)} sample={sentences[:3]}")

        print(f"total_sentences: {total_sentences}")
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
