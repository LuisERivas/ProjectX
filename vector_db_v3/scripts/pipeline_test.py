from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def emit_usage_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def resolve_cli_binary(build_dir: Path) -> Path | None:
    candidates = [build_dir / "vectordb_v3_cli.exe", build_dir / "vectordb_v3_cli"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step-1 scaffold for vector_db_v3 synthetic pipeline testing.",
    )
    parser.add_argument(
        "--build-dir",
        required=True,
        help="Path to build directory containing vectordb_v3_cli(.exe).",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Optional data directory for future pipeline artifacts.",
    )
    parser.add_argument(
        "--embedding-count",
        required=True,
        type=int,
        help="Required embedding count (>0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for future bulk ingest path (>0).",
    )
    parser.add_argument(
        "--results-out",
        default="",
        help="Optional output JSON path; default is <data-dir>/pipeline_test_results.json.",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Preserve test data directory after future pipeline runs.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.embedding_count <= 0:
        return emit_usage_error("--embedding-count must be > 0")
    if args.batch_size <= 0:
        return emit_usage_error("--batch-size must be > 0")

    build_dir = Path(args.build_dir).resolve()
    cli_binary = resolve_cli_binary(build_dir)
    if cli_binary is None:
        return emit_usage_error(f"missing vectordb_v3_cli binary in --build-dir: {build_dir}")

    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        data_dir = Path(tempfile.gettempdir()) / "vectordb_v3_pipeline_test"

    if args.results_out:
        results_out = Path(args.results_out).resolve()
    else:
        results_out = data_dir / "pipeline_test_results.json"

    payload = {
        "status": "ok",
        "command": "pipeline-test-skeleton",
        "build_dir": str(build_dir),
        "cli_binary": str(cli_binary),
        "data_dir": str(data_dir),
        "embedding_count": int(args.embedding_count),
        "batch_size": int(args.batch_size),
        "results_out": str(results_out),
        "keep_data": bool(args.keep_data),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

