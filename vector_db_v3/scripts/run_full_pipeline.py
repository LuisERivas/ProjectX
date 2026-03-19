from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> int:
    print(f"[RUN] {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(cwd))
    return int(completed.returncode)


def main() -> int:
    script_path = Path(__file__).resolve()
    v3_root = script_path.parents[1]
    repo_root = v3_root.parent

    build_dir = v3_root / "build"
    data_dir = Path("/tmp/v3_auto_full_pipeline")
    embedding_count = 10000

    configure_cmd = ["cmake", "-S", str(v3_root), "-B", str(build_dir)]
    build_cmd = ["cmake", "--build", str(build_dir), "-j"]
    run_pipeline_cmd = [
        sys.executable,
        str(v3_root / "scripts" / "pipeline_test.py"),
        "--build-dir",
        str(build_dir),
        "--data-dir",
        str(data_dir),
        "--embedding-count",
        str(embedding_count),
        "--input-format",
        "bin",
        "--run-full-pipeline",
        "--orchestration-mode",
        "composite",
    ]

    rc = run(configure_cmd, repo_root)
    if rc != 0:
        print("error: cmake configure failed", file=sys.stderr)
        return rc

    rc = run(build_cmd, repo_root)
    if rc != 0:
        print("error: build failed", file=sys.stderr)
        return rc

    rc = run(run_pipeline_cmd, repo_root)
    if rc != 0:
        print("error: full pipeline run failed", file=sys.stderr)
        return rc

    print("[OK] Full pipeline completed.")
    print(f"[INFO] Data dir: {data_dir}")
    print(f"[INFO] Results: {data_dir / 'pipeline_test_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
