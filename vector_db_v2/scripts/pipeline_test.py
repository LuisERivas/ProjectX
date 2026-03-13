from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_stage(name: str, cmd: list[str], cwd: Path) -> str:
    t0 = time.time()
    print(f"[stage_start] {name}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr.rstrip())
        raise RuntimeError(f"stage failed: {name} ({elapsed:.3f}s)")
    print(f"[stage_end] {name}: {elapsed:.3f}s")
    return proc.stdout


def generate_payload(path: Path, rows: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(1, rows + 1):
            vals = [f"{((i % 97) * 0.001) + ((j % 13) * 0.01):.6f}" for j in range(1024)]
            row = {"embedding_id": i, "vec_csv": ",".join(vals)}
            f.write(json.dumps(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build_dir = root / "build"
    data_dir = root / "tmp_pipeline_data"
    payload = root / "tmp_pipeline_insert.jsonl"
    cli = build_dir / "vectordb_v2_cli.exe"
    if not cli.exists():
        cli = build_dir / "vectordb_v2_cli"

    if data_dir.exists():
        shutil.rmtree(data_dir)
    if payload.exists():
        payload.unlink()

    if not args.skip_build:
        run_stage("cmake configure", ["cmake", "-S", str(root), "-B", str(build_dir)], root)
        run_stage("cmake build", ["cmake", "--build", str(build_dir)], root)
        run_stage("ctest", ["ctest", "--test-dir", str(build_dir), "--output-on-failure"], root)

    if not cli.exists():
        raise RuntimeError("missing vectordb_v2_cli after build")

    generate_payload(payload, args.rows)

    run_stage("init", [str(cli), "init", "--path", str(data_dir)], root)
    run_stage("bulk-insert", [str(cli), "bulk-insert", "--path", str(data_dir), "--input", str(payload)], root)
    run_stage("checkpoint", [str(cli), "checkpoint", "--path", str(data_dir)], root)
    run_stage("build-top-clusters", [str(cli), "build-top-clusters", "--path", str(data_dir)], root)
    run_stage("build-mid-layer-clusters", [str(cli), "build-mid-layer-clusters", "--path", str(data_dir)], root)
    run_stage("build-lower-layer-clusters", [str(cli), "build-lower-layer-clusters", "--path", str(data_dir)], root)
    run_stage("build-final-layer-clusters", [str(cli), "build-final-layer-clusters", "--path", str(data_dir)], root)
    cstats = run_stage("cluster-stats", [str(cli), "cluster-stats", "--path", str(data_dir)], root)
    chealth = run_stage("cluster-health", [str(cli), "cluster-health", "--path", str(data_dir)], root)

    out = {
        "data_dir": str(data_dir),
        "cluster_stats": json.loads(cstats),
        "cluster_health": json.loads(chealth),
        "artifacts": {
            "mid": str(data_dir / "clusters" / "current" / "mid_layer_clustering" / "MID_LAYER_CLUSTERING.json"),
            "lower": str(data_dir / "clusters" / "current" / "lower_layer_clustering" / "LOWER_LAYER_CLUSTERING.json"),
            "final": str(data_dir / "clusters" / "current" / "final_layer_clustering" / "FINAL_LAYER_DBSCAN.json"),
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
