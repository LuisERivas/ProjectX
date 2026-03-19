from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> int:
    print(f"[RUN] {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(cwd), env=env)
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and run the fully optimized vector_db_v3 pipeline with stats."
    )
    parser.add_argument(
        "--build-dir",
        default="",
        help="Optional build directory. Default: <vector_db_v3>/build",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Optional pipeline data directory. Default: temp/v3_auto_full_pipeline_optimized",
    )
    parser.add_argument(
        "--embedding-count",
        type=int,
        default=10000,
        help="Synthetic embedding count (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for composite run (default: 1234).",
    )
    parser.add_argument(
        "--input-format",
        choices=["bin", "jsonl"],
        default="bin",
        help="Synthetic ingest input format (default: bin).",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip cmake configure/build and run pipeline only.",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep generated data/artifacts after pipeline run.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    script_path = Path(__file__).resolve()
    v3_root = script_path.parents[1]
    repo_root = v3_root.parent

    build_dir = Path(args.build_dir).resolve() if args.build_dir else (v3_root / "build")
    data_dir = (
        Path(args.data_dir).resolve()
        if args.data_dir
        else Path(tempfile.gettempdir()) / "v3_auto_full_pipeline_optimized"
    )
    embedding_count = int(args.embedding_count)

    if embedding_count <= 0:
        print("error: --embedding-count must be > 0", file=sys.stderr)
        return 2

    configure_cmd = ["cmake", "-S", str(v3_root), "-B", str(build_dir)]
    build_cmd = ["cmake", "--build", str(build_dir), "-j"]

    pipeline_cmd = [
        sys.executable,
        str(v3_root / "scripts" / "pipeline_test.py"),
        "--build-dir",
        str(build_dir),
        "--data-dir",
        str(data_dir),
        "--embedding-count",
        str(embedding_count),
        "--input-format",
        args.input_format,
        "--run-full-pipeline",
        "--orchestration-mode",
        "composite",
        "--with-cluster-stats",
        "--seed",
        str(args.seed),
    ]
    if args.keep_data:
        pipeline_cmd.append("--keep-data")

    # Enable all known optimization toggles without requiring strict hardware-only modes.
    env = dict(os.environ)
    env.update(
        {
            "VECTOR_DB_V3_INGEST_ASYNC_MODE": "1",
            "VECTOR_DB_V3_INGEST_PINNED": "1",
            "VECTOR_DB_V3_WAL_COMMIT_POLICY": "batch_group_commit",
            "VECTOR_DB_V3_POST_INGEST_CHECKPOINT": "1",
            "VECTOR_DB_V3_KMEANS_BACKEND": "auto",
            "VECTOR_DB_V3_KMEANS_PRECISION": "tensor_fp16",
            "VECTOR_DB_V3_INTERNAL_SHARD_MODE": "auto",
            "VECTOR_DB_V3_INTERNAL_SHARD_REPAIR": "regenerate",
            "VECTOR_DB_V3_GPU_RESIDENCY_MODE": "auto",
        }
    )

    print("[INFO] Optimizations enabled:")
    for key in [
        "VECTOR_DB_V3_INGEST_ASYNC_MODE",
        "VECTOR_DB_V3_INGEST_PINNED",
        "VECTOR_DB_V3_WAL_COMMIT_POLICY",
        "VECTOR_DB_V3_POST_INGEST_CHECKPOINT",
        "VECTOR_DB_V3_KMEANS_BACKEND",
        "VECTOR_DB_V3_KMEANS_PRECISION",
        "VECTOR_DB_V3_INTERNAL_SHARD_MODE",
        "VECTOR_DB_V3_INTERNAL_SHARD_REPAIR",
        "VECTOR_DB_V3_GPU_RESIDENCY_MODE",
    ]:
        print(f"  - {key}={env[key]}")

    if not args.skip_build:
        rc = run(configure_cmd, repo_root, env=env)
        if rc != 0:
            print("error: cmake configure failed", file=sys.stderr)
            return rc

        rc = run(build_cmd, repo_root, env=env)
        if rc != 0:
            print("error: build failed", file=sys.stderr)
            return rc

    rc = run(pipeline_cmd, repo_root, env=env)
    if rc != 0:
        print("error: optimized full pipeline run failed", file=sys.stderr)
        return rc

    print("[OK] Optimized full pipeline completed.")
    print(f"[INFO] Data dir: {data_dir}")
    print(f"[INFO] Results: {data_dir / 'pipeline_test_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

