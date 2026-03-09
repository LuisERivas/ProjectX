from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def make_vec(seed: int) -> str:
    vals = [0.0] * 1024
    base = seed % 128
    vals[base] = 0.8
    vals[(base + 1) % 1024] = 0.6
    return ",".join(f"{v:.6f}" for v in vals)


def main() -> int:
    build_dir = Path("build")
    bin_name = "vectordb_cli.exe" if sys.platform.startswith("win") else "vectordb_cli"
    cli = build_dir / bin_name
    if not cli.exists():
        raise RuntimeError(f"missing CLI binary at {cli}; build first with cmake")

    data_dir = Path("benchmark_data")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    run([str(cli), "init", "--path", str(data_dir)])
    total = 256
    for i in range(total):
        run([str(cli), "insert", "--path", str(data_dir), "--id", str(10000 + i), "--vec", make_vec(i), "--meta", '{"kind":"bench"}'])

    t0 = time.perf_counter()
    run([str(cli), "build-initial-clusters", "--path", str(data_dir), "--seed", "42"])
    t1 = time.perf_counter()
    elapsed = t1 - t0
    stats = json.loads(run([str(cli), "cluster-stats", "--path", str(data_dir)]))
    health = json.loads(run([str(cli), "cluster-health", "--path", str(data_dir)]))

    print(json.dumps(
        {
            "vectors": total,
            "seconds": round(elapsed, 6),
            "vectors_per_second": round(total / max(elapsed, 1e-9), 3),
            "used_cuda": stats.get("used_cuda", False),
            "tensor_core_enabled": stats.get("tensor_core_enabled", False),
            "gpu_backend": stats.get("gpu_backend", "unknown"),
            "scoring_ms_total": stats.get("scoring_ms_total", 0.0),
            "scoring_calls": stats.get("scoring_calls", 0),
            "stability_passed": health.get("passed", False),
        },
        indent=2,
    ))
    shutil.rmtree(data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

