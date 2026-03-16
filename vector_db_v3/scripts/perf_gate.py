from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return proc.returncode, proc.stdout, proc.stderr, elapsed_ms


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Soft performance gate harness for evidence collection.")
    parser.add_argument("--build-dir", default="vector_db_v3/build")
    parser.add_argument("--mode", choices=["evidence"], default="evidence")
    parser.add_argument("--profile", choices=["minimum"], default="minimum")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=3)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = (repo_root.parent / build_dir).resolve()
    if not build_dir.exists():
        print(f"error: build dir not found: {build_dir}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "gate_evidence" / "perf_ad_hoc")
    if not out_dir.is_absolute():
        out_dir = (repo_root.parent / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "perf.log"

    ctest_cmd = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
        "-R",
        "vectordb_v3_exact_search_tests|vectordb_v3_top_layer_artifacts_tests|vectordb_v3_mid_layer_artifacts_tests|vectordb_v3_final_layer_artifacts_tests",
    ]
    env = os.environ.copy()
    env["VECTOR_DB_V3_COMPLIANCE_PROFILE"] = "pass"

    elapsed: list[float] = []
    failures = 0
    with log_path.open("w", encoding="utf-8") as logf:
        for i in range(args.warmup_runs):
            code, out, err, ms = run_command(ctest_cmd, cwd=repo_root.parent, env=env)
            logf.write(f"[warmup {i+1}] exit={code} elapsed_ms={ms:.3f}\n")
            if out:
                logf.write(out + ("" if out.endswith("\n") else "\n"))
            if err:
                logf.write("--- stderr ---\n")
                logf.write(err + ("" if err.endswith("\n") else "\n"))
            if code != 0:
                failures += 1

        for i in range(args.measure_runs):
            code, out, err, ms = run_command(ctest_cmd, cwd=repo_root.parent, env=env)
            logf.write(f"[measure {i+1}] exit={code} elapsed_ms={ms:.3f}\n")
            if out:
                logf.write(out + ("" if out.endswith("\n") else "\n"))
            if err:
                logf.write("--- stderr ---\n")
                logf.write(err + ("" if err.endswith("\n") else "\n"))
            elapsed.append(ms)
            if code != 0:
                failures += 1

    median_ms = statistics.median(elapsed) if elapsed else 0.0
    p95_ms = sorted(elapsed)[max(0, min(len(elapsed) - 1, int(len(elapsed) * 0.95) - 1))] if elapsed else 0.0
    variance_ratio = (statistics.pstdev(elapsed) / median_ms) if elapsed and median_ms > 0 else 0.0

    status = "pass_soft"
    notes: list[str] = []
    if failures > 0:
        status = "warn"
        notes.append("one or more performance harness runs failed")
    if variance_ratio > 0.35:
        status = "warn"
        notes.append("high runtime variance detected")

    metrics = {
        "mode": args.mode,
        "profile": args.profile,
        "warmup_runs": args.warmup_runs,
        "measure_runs": args.measure_runs,
        "elapsed_ms": elapsed,
        "median_ms": round(median_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "variance_ratio": round(variance_ratio, 5),
        "failures": failures,
    }
    write_json(out_dir / "perf_metrics.json", metrics)
    result = {
        "gate_id": "G4",
        "status": status,
        "notes": notes,
        "artifact_paths": [str(out_dir / "perf_metrics.json"), str(log_path)],
    }
    write_json(out_dir / "result.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

