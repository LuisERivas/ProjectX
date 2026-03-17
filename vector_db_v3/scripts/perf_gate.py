from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return proc.returncode, proc.stdout, proc.stderr, elapsed_ms
    except FileNotFoundError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return 127, "", str(exc), elapsed_ms


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * p))
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


def summarize_runs(elapsed: list[float]) -> dict:
    median_ms = statistics.median(elapsed) if elapsed else 0.0
    p95_ms = percentile(elapsed, 0.95)
    variance_ratio = (statistics.pstdev(elapsed) / median_ms) if elapsed and median_ms > 0 else 0.0
    return {
        "samples_ms": [round(v, 3) for v in elapsed],
        "median_ms": round(median_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "variance_ratio": round(variance_ratio, 5),
    }


def run_perf_probe(
    ctest_cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    warmup_runs: int,
    measure_runs: int,
    log_path: Path,
) -> tuple[dict, int]:
    elapsed: list[float] = []
    failures = 0
    raw_runs: list[dict] = []
    with log_path.open("w", encoding="utf-8") as logf:
        for i in range(warmup_runs):
            code, out, err, ms = run_command(ctest_cmd, cwd=cwd, env=env)
            row = {"phase": "warmup", "run_index": i + 1, "exit_code": code, "elapsed_ms": round(ms, 3)}
            raw_runs.append(row)
            logf.write(f"[warmup {i+1}] exit={code} elapsed_ms={ms:.3f}\n")
            if out:
                logf.write(out + ("" if out.endswith("\n") else "\n"))
            if err:
                logf.write("--- stderr ---\n")
                logf.write(err + ("" if err.endswith("\n") else "\n"))
            if code != 0:
                failures += 1

        for i in range(measure_runs):
            code, out, err, ms = run_command(ctest_cmd, cwd=cwd, env=env)
            row = {"phase": "measure", "run_index": i + 1, "exit_code": code, "elapsed_ms": round(ms, 3)}
            raw_runs.append(row)
            logf.write(f"[measure {i+1}] exit={code} elapsed_ms={ms:.3f}\n")
            if out:
                logf.write(out + ("" if out.endswith("\n") else "\n"))
            if err:
                logf.write("--- stderr ---\n")
                logf.write(err + ("" if err.endswith("\n") else "\n"))
            elapsed.append(ms)
            if code != 0:
                failures += 1
    summary = summarize_runs(elapsed)
    summary["raw_runs"] = raw_runs
    return summary, failures


def build_thresholds_from_baseline(baseline: dict, median_headroom: float, p95_headroom: float) -> dict:
    perf = baseline["measured"]
    return {
        "policy_name": "baseline_plus_headroom",
        "median_headroom_ratio": median_headroom,
        "p95_headroom_ratio": p95_headroom,
        "variance_ratio_max": 0.35,
        "max_failures": 0,
        "limits_ms": {
            "median_ms_max": round(perf["median_ms"] * (1.0 + median_headroom), 3),
            "p95_ms_max": round(perf["p95_ms"] * (1.0 + p95_headroom), 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Performance gate harness with evidence, baseline, and hard modes.")
    parser.add_argument("--build-dir", default="vector_db_v3/build")
    parser.add_argument("--mode", choices=["evidence", "baseline", "hard"], default="evidence")
    parser.add_argument("--profile", choices=["minimum", "jetson_orin"], default="minimum")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=3)
    parser.add_argument("--baseline-ref", default="")
    parser.add_argument("--thresholds-file", default="")
    parser.add_argument("--median-headroom-ratio", type=float, default=0.10)
    parser.add_argument("--p95-headroom-ratio", type=float, default=0.15)
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
    raw_runs_path = out_dir / "raw_runs.json"
    environment_path = out_dir / "environment.json"
    perf_metrics_path = out_dir / "perf_metrics.json"
    result_path = out_dir / "result.json"

    default_thresholds = repo_root / "contracts" / "perf_thresholds_jetson_orin.json"
    thresholds_file = Path(args.thresholds_file) if args.thresholds_file else default_thresholds
    if not thresholds_file.is_absolute():
        thresholds_file = (repo_root.parent / thresholds_file).resolve()

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

    env_payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": args.mode,
        "profile": args.profile,
        "cwd": str(repo_root.parent),
        "build_dir": str(build_dir),
        "ctest_cmd": ctest_cmd,
        "warmup_runs": args.warmup_runs,
        "measure_runs": args.measure_runs,
    }
    write_json(environment_path, env_payload)

    measured, failures = run_perf_probe(
        ctest_cmd=ctest_cmd,
        cwd=repo_root.parent,
        env=env,
        warmup_runs=args.warmup_runs,
        measure_runs=args.measure_runs,
        log_path=log_path,
    )
    write_json(raw_runs_path, {"runs": measured["raw_runs"]})

    metrics = {
        "mode": args.mode,
        "profile": args.profile,
        "baseline_ref": args.baseline_ref if args.baseline_ref else "none",
        "thresholds_file": str(thresholds_file),
        "measured": {
            "median_ms": measured["median_ms"],
            "p95_ms": measured["p95_ms"],
            "variance_ratio": measured["variance_ratio"],
            "samples_ms": measured["samples_ms"],
        },
        "harness": {
            "warmup_runs": args.warmup_runs,
            "measure_runs": args.measure_runs,
            "failures": failures,
        },
    }
    write_json(perf_metrics_path, metrics)

    status = "pass_soft"
    notes: list[str] = []
    failed_metrics: list[str] = []
    exit_code = 0

    if args.mode == "evidence":
        if failures > 0:
            status = "warn_nonblocking"
            notes.append("one or more performance harness runs failed")
        if measured["variance_ratio"] > 0.35:
            status = "warn_nonblocking"
            notes.append("high runtime variance detected")
    elif args.mode == "baseline":
        if failures > 0:
            status = "fail_env"
            notes.append("baseline capture failed: one or more harness runs failed")
            exit_code = 2
            result = {
                "gate_id": "G4",
                "status": status,
                "mode": args.mode,
                "profile": args.profile,
                "failed_metrics": failed_metrics,
                "exit_code": exit_code,
                "notes": notes,
                "artifact_paths": [
                    str(perf_metrics_path),
                    str(raw_runs_path),
                    str(environment_path),
                    str(log_path),
                ],
            }
            write_json(result_path, result)
            print(json.dumps(result, indent=2))
            return exit_code

        baseline_payload = {
            "profile": args.profile,
            "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "measured": metrics["measured"],
        }
        baseline_path = out_dir / "baseline_metrics.json"
        write_json(baseline_path, baseline_payload)
        thresholds = build_thresholds_from_baseline(
            baseline_payload,
            args.median_headroom_ratio,
            args.p95_headroom_ratio,
        )
        thresholds["profile"] = args.profile
        write_json(thresholds_file, thresholds)
        status = "pass"
        notes.append("baseline and thresholds written")
    else:
        if not thresholds_file.exists():
            status = "fail_env"
            notes.append(f"thresholds file missing: {thresholds_file}")
            exit_code = 2
        else:
            thresholds = read_json(thresholds_file)
            limits = thresholds.get("limits_ms", {})
            median_limit = float(limits.get("median_ms_max", 0.0))
            p95_limit = float(limits.get("p95_ms_max", 0.0))
            variance_limit = float(thresholds.get("variance_ratio_max", 0.35))
            max_failures = int(thresholds.get("max_failures", 0))

            if failures > max_failures:
                failed_metrics.append(f"harness_failures>{max_failures}")
            if measured["median_ms"] > median_limit:
                failed_metrics.append("median_ms")
            if measured["p95_ms"] > p95_limit:
                failed_metrics.append("p95_ms")
            if measured["variance_ratio"] > variance_limit:
                failed_metrics.append("variance_ratio")

            metrics["threshold_policy"] = {
                "source": str(thresholds_file),
                "limits_ms": {"median_ms_max": median_limit, "p95_ms_max": p95_limit},
                "variance_ratio_max": variance_limit,
                "max_failures": max_failures,
            }
            write_json(perf_metrics_path, metrics)

            if failed_metrics:
                status = "fail_threshold"
                notes.append("hard limits exceeded")
                exit_code = 1
            else:
                status = "pass"
                exit_code = 0

    result = {
        "gate_id": "G4",
        "status": status,
        "mode": args.mode,
        "profile": args.profile,
        "failed_metrics": failed_metrics,
        "exit_code": exit_code,
        "notes": notes,
        "artifact_paths": [
            str(perf_metrics_path),
            str(raw_runs_path),
            str(environment_path),
            str(log_path),
        ],
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

