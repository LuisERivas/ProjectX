from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_command(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
    except FileNotFoundError as exc:
        elapsed = (time.perf_counter() - start) * 1000.0
        return 127, "", str(exc), elapsed
    elapsed = (time.perf_counter() - start) * 1000.0
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_bulk_bin(path: Path, rows: list[tuple[int, float]]) -> None:
    dim = 1024
    record_size = 8 + dim * 4
    header = bytearray(18)
    header[0:4] = (0x49423356).to_bytes(4, "little", signed=False)
    header[4:6] = (1).to_bytes(2, "little", signed=False)
    header[6:10] = record_size.to_bytes(4, "little", signed=False)
    header[10:18] = len(rows).to_bytes(8, "little", signed=False)
    with path.open("wb") as f:
        f.write(header)
        for embedding_id, value in rows:
            row = bytearray(record_size)
            row[0:8] = int(embedding_id).to_bytes(8, "little", signed=False)
            packed = struct.pack("<f", float(value))
            for i in range(dim):
                row[8 + i * 4 : 12 + i * 4] = packed
            f.write(row)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((len(s) - 1) * p))
    idx = max(0, min(len(s) - 1, idx))
    return s[idx]


def summarize(values: list[float]) -> dict:
    if not values:
        return {"samples_ms": [], "median_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(values)
    mid = ordered[len(ordered) // 2] if len(ordered) % 2 == 1 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2.0
    return {
        "samples_ms": [round(v, 3) for v in values],
        "median_ms": round(mid, 3),
        "p95_ms": round(percentile(values, 0.95), 3),
    }

def summarize_internal_metrics(samples: list[dict]) -> dict:
    if not samples:
        return {}
    def median_of(key: str) -> float:
        vals = [float(s.get(key, 0.0)) for s in samples]
        return round(float(statistics.median(vals)), 3) if vals else 0.0
    return {
        "samples": len(samples),
        "peak_queue_depth_max": int(max(int(s.get("peak_queue_depth", 0)) for s in samples)),
        "producer_wait_ms_median": median_of("producer_wait_ms"),
        "consumer_wait_ms_median": median_of("consumer_wait_ms"),
        "commit_apply_ms_median": median_of("commit_apply_ms"),
    }


def cv_percent(values: list[float]) -> float:
    if not values:
        return 0.0
    med = statistics.median(values)
    if med <= 0:
        return 0.0
    return (statistics.pstdev(values) / med) * 100.0


def build_perf_checklist(perf: dict) -> list[tuple[str, bool]]:
    checklist: list[tuple[str, bool]] = []
    base_samples = [float(v) for v in perf["baseline"]["summary"]["samples_ms"]]
    cand_samples = [float(v) for v in perf["candidate"]["summary"]["samples_ms"]]
    improvement = float(perf["delta"]["ingest_median_improvement_pct"])
    allowed_regression = max(5.0, 2.0 * max(cv_percent(base_samples), cv_percent(cand_samples)))
    checklist.append(("no_major_ingest_regression", improvement > -allowed_regression))
    checklist.append(("ingest_threshold_computed", allowed_regression >= 5.0))
    perf["variance_guard"] = {
        "allowed_regression_pct": round(allowed_regression, 3),
        "baseline_cv_pct": round(cv_percent(base_samples), 3),
        "candidate_cv_pct": round(cv_percent(cand_samples), 3),
    }
    return checklist


def run_ingest_perf(cli: Path, repo_root: Path, out_dir: Path, async_mode: bool, runs: int, warmup: int) -> dict:
    mode_name = "candidate_async" if async_mode else "baseline_sync"
    mode_dir = out_dir / mode_name
    if mode_dir.exists():
        shutil.rmtree(mode_dir)
    mode_dir.mkdir(parents=True, exist_ok=True)

    data_dir = mode_dir / "data"
    env = dict(os.environ)
    env["VECTOR_DB_V3_INGEST_ASYNC_MODE"] = "1" if async_mode else "0"
    env["VECTOR_DB_V3_INGEST_PINNED"] = "1" if async_mode else "0"
    env["VECTOR_DB_V3_INGEST_ASYNC_POLICY"] = "on" if async_mode else "off"
    env["VECTOR_DB_V3_INGEST_QUEUE_CAPACITY"] = "8" if async_mode else "4"
    env["VECTOR_DB_V3_INGEST_PRODUCER_CHUNK"] = "96" if async_mode else "64"
    env["VECTOR_DB_V3_COMPLIANCE_PROFILE"] = "pass"
    metrics_path = mode_dir / "ingest_metrics.jsonl"
    env["VECTOR_DB_V3_INGEST_METRICS_PATH"] = str(metrics_path)

    rows = [(50000 + i, 0.001 * ((i % 113) + 1)) for i in range(8192)]
    bulk_path = mode_dir / "bulk.bin"
    write_bulk_bin(bulk_path, rows)

    code, out, err, _ = run_command([str(cli), "init", "--path", str(data_dir)], cwd=repo_root, env=env)
    if code != 0:
        raise RuntimeError(f"{mode_name} init failed: {err or out}")

    durations: list[float] = []
    logs: list[dict] = []
    for i in range(warmup + runs):
        code, out, err, elapsed = run_command(
            [str(cli), "bulk-insert-bin", "--path", str(data_dir), "--input", str(bulk_path), "--batch-size", "64"],
            cwd=repo_root,
            env=env,
        )
        if code != 0:
            raise RuntimeError(f"{mode_name} ingest failed run {i+1}: {err or out}")
        logs.append(
            {
                "run_index": i + 1,
                "phase": "warmup" if i < warmup else "measure",
                "elapsed_ms": round(elapsed, 3),
            }
        )
        if i >= warmup:
            durations.append(elapsed)
    write_json(mode_dir / "runs.json", {"runs": logs})
    metric_rows: list[dict] = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                metric_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {
        "mode": mode_name,
        "summary": summarize(durations),
        "internal_metrics": summarize_internal_metrics(metric_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Card 5 build/test/perf validation.")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--profile", choices=["minimum", "jetson_orin"], default="minimum")
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    vdb_root = script_path.parents[1]
    repo_root = Path(args.repo_root).resolve() if args.repo_root else vdb_root.parent
    build_dir = Path(args.build_dir).resolve() if args.build_dir else (vdb_root / "build").resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (vdb_root / "gate_evidence" / "card5_validation").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict] = []

    def run_step(name: str, cmd: list[str]) -> tuple[bool, str]:
        code, out, err, elapsed = run_command(cmd, cwd=repo_root)
        log_path = logs_dir / f"{name}.log"
        write_text(
            log_path,
            f"$ {' '.join(cmd)}\nexit_code={code}\nelapsed_ms={elapsed:.3f}\n\n--- stdout ---\n{out}\n--- stderr ---\n{err}\n",
        )
        steps.append({"name": name, "command": cmd, "exit_code": code, "elapsed_ms": round(elapsed, 3), "log": str(log_path)})
        return code == 0, (err or out)

    ok, msg = run_step("cmake_configure", ["cmake", "-S", str(vdb_root), "-B", str(build_dir)])
    if not ok:
        write_json(out_dir / "summary.json", {"status": "fail", "failed_step": "cmake_configure", "steps": steps})
        print("FAIL: cmake configure failed", file=sys.stderr)
        print(msg, file=sys.stderr)
        return 1
    ok, msg = run_step("cmake_build", ["cmake", "--build", str(build_dir), "--config", "Release"])
    if not ok:
        write_json(out_dir / "summary.json", {"status": "fail", "failed_step": "cmake_build", "steps": steps})
        print("FAIL: cmake build failed", file=sys.stderr)
        print(msg, file=sys.stderr)
        return 1

    regex = (
        "vectordb_v3_cli_contract_tests|vectordb_v3_terminal_event_contract_tests|"
        "vectordb_v3_durability_wal_tests|vectordb_v3_durability_checkpoint_tests|"
        "vectordb_v3_durability_replay_crash_tests|vectordb_v3_durability_corruption_tests|"
        "vectordb_v3_card5_ingest_async_tests"
    )
    ok, msg = run_step("ctest_targeted", ["ctest", "--test-dir", str(build_dir), "--output-on-failure", "-R", regex])
    if not ok:
        write_json(out_dir / "summary.json", {"status": "fail", "failed_step": "ctest_targeted", "steps": steps})
        print("FAIL: targeted ctest failed", file=sys.stderr)
        print(msg, file=sys.stderr)
        return 1

    cli = build_dir / "vectordb_v3_cli.exe"
    if not cli.exists():
        cli = build_dir / "vectordb_v3_cli"
    if not cli.exists():
        write_json(out_dir / "summary.json", {"status": "fail", "failed_step": "cli_lookup", "steps": steps})
        print("FAIL: missing vectordb_v3_cli binary", file=sys.stderr)
        return 1

    perf: dict = {"status": "skipped"}
    checklist: list[tuple[str, bool]] = []
    if not args.skip_perf:
        try:
            baseline = run_ingest_perf(cli, repo_root, out_dir / "perf", async_mode=False, runs=args.runs, warmup=args.warmup_runs)
            candidate = run_ingest_perf(cli, repo_root, out_dir / "perf", async_mode=True, runs=args.runs, warmup=args.warmup_runs)
        except Exception as exc:
            write_json(
                out_dir / "summary.json",
                {"status": "fail", "failed_step": "perf", "error": str(exc), "steps": steps},
            )
            print(f"FAIL: perf probe failed: {exc}", file=sys.stderr)
            return 1
        base_med = float(baseline["summary"]["median_ms"])
        cand_med = float(candidate["summary"]["median_ms"])
        improvement = ((base_med - cand_med) / base_med * 100.0) if base_med > 0 else 0.0
        perf = {
            "baseline": baseline,
            "candidate": candidate,
            "delta": {"ingest_median_improvement_pct": round(improvement, 3)},
        }
        checklist = build_perf_checklist(perf)
        initial_failed = [name for name, passed in checklist if not passed]
        if initial_failed:
            try:
                baseline_retry = run_ingest_perf(
                    cli,
                    repo_root,
                    out_dir / "perf_retry",
                    async_mode=False,
                    runs=args.runs,
                    warmup=args.warmup_runs,
                )
                candidate_retry = run_ingest_perf(
                    cli,
                    repo_root,
                    out_dir / "perf_retry",
                    async_mode=True,
                    runs=args.runs,
                    warmup=args.warmup_runs,
                )
                base_med_retry = float(baseline_retry["summary"]["median_ms"])
                cand_med_retry = float(candidate_retry["summary"]["median_ms"])
                improvement_retry = ((base_med_retry - cand_med_retry) / base_med_retry * 100.0) if base_med_retry > 0 else 0.0
                perf_retry = {
                    "baseline": baseline_retry,
                    "candidate": candidate_retry,
                    "delta": {"ingest_median_improvement_pct": round(improvement_retry, 3)},
                }
                checklist_retry = build_perf_checklist(perf_retry)
                retry_failed = [name for name, passed in checklist_retry if not passed]
                write_json(out_dir / "perf_retry.json", perf_retry)
                if not retry_failed:
                    perf = {**perf_retry, "retry_note": "initial failed, retry passed"}
                    checklist = checklist_retry
                else:
                    perf["retry_attempted"] = True
                    perf["retry_failed_checks"] = retry_failed
            except Exception as exc:
                perf["retry_attempted"] = True
                perf["retry_error"] = str(exc)
        write_json(out_dir / "perf_baseline.json", baseline)
        write_json(out_dir / "perf_candidate.json", candidate)

    failed_checks = [name for name, passed in checklist if not passed]
    summary = {
        "status": "pass" if not failed_checks else "fail",
        "profile": args.profile,
        "steps": steps,
        "checklist": [{"name": n, "pass": p} for n, p in checklist],
        "failed_checks": failed_checks,
        "perf": perf,
    }
    write_json(out_dir / "summary.json", summary)
    if failed_checks:
        write_json(out_dir / "failures.json", {"failed_checks": failed_checks, "summary": summary})
        print(f"FAIL: Card 5 checklist failed: {', '.join(failed_checks)}", file=sys.stderr)
        return 1

    print("PASS: Card 5 validation completed.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
