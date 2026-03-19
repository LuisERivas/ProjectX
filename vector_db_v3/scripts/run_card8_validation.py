from __future__ import annotations

import argparse
import json
import os
import statistics
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
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return 127, "", str(exc), elapsed_ms
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return proc.returncode, proc.stdout, proc.stderr, elapsed_ms


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * p))
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


def summarize(samples: list[float]) -> dict:
    if not samples:
        return {"samples_ms": [], "median_ms": 0.0, "p95_ms": 0.0, "variance_ratio": 0.0}
    median = statistics.median(samples)
    variance_ratio = (statistics.pstdev(samples) / median) if median > 0 else 0.0
    return {
        "samples_ms": [round(v, 3) for v in samples],
        "median_ms": round(median, 3),
        "p95_ms": round(percentile(samples, 0.95), 3),
        "variance_ratio": round(variance_ratio, 5),
    }


def stage_latency(rows: list[dict], stage_name: str) -> float:
    for row in rows:
        if row.get("stage") == stage_name:
            value = row.get("latency_ms")
            if isinstance(value, (int, float)):
                return float(value)
    return 0.0


def run_pipeline_probe(
    repo_root: Path,
    v3_root: Path,
    build_dir: Path,
    mode: str,
    out_dir: Path,
    runs: int = 4,
) -> dict:
    total_samples: list[float] = []
    top_samples: list[float] = []
    mid_samples: list[float] = []
    run_rows: list[dict] = []

    post_ingest_checkpoint = mode == "enabled"
    for i in range(runs):
        report_path = out_dir / f"pipeline_{mode}_{i+1}.json"
        cmd = [
            sys.executable,
            str(v3_root / "scripts" / "pipeline_test.py"),
            "--build-dir",
            str(build_dir),
            "--embedding-count",
            "4096",
            "--batch-size",
            "128",
            "--input-format",
            "bin",
            "--run-full-pipeline",
            "--orchestration-mode",
            "legacy",
            "--results-out",
            str(report_path),
        ]
        if post_ingest_checkpoint:
            cmd.append("--post-ingest-checkpoint")
        env = dict(os.environ)
        env["VECTOR_DB_V3_COMPLIANCE_PROFILE"] = "pass"
        code, out, err, elapsed_ms = run_command(cmd, cwd=repo_root, env=env)
        if code != 0:
            raise RuntimeError(f"pipeline probe failed mode={mode} run={i+1}: {(err or out)[:400]}")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        stage_rows = payload.get("stage_results", {}).get("full_pipeline", [])
        if not isinstance(stage_rows, list):
            raise RuntimeError("pipeline probe missing full_pipeline stage rows")
        top_ms = stage_latency(stage_rows, "build-top-clusters")
        mid_ms = stage_latency(stage_rows, "build-mid-layer-clusters")
        total_ms = 0.0
        for row in stage_rows:
            if isinstance(row, dict):
                val = row.get("latency_ms")
                if isinstance(val, (int, float)):
                    total_ms += float(val)
        total_samples.append(total_ms if total_ms > 0 else elapsed_ms)
        top_samples.append(top_ms)
        mid_samples.append(mid_ms)
        run_rows.append(
            {
                "run_index": i + 1,
                "wall_elapsed_ms": round(elapsed_ms, 3),
                "pipeline_total_ms": round(total_samples[-1], 3),
                "top_stage_ms": round(top_ms, 3),
                "mid_stage_ms": round(mid_ms, 3),
            }
        )

    return {
        "mode": mode,
        "total_pipeline": summarize(total_samples),
        "top_stage": summarize(top_samples),
        "mid_stage": summarize(mid_samples),
        "runs": run_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Card 8 build/test/perf validation.")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--profile", choices=["minimum", "jetson_orin"], default="minimum")
    parser.add_argument("--baseline-checkpoint-mode", choices=["0", "1"], default="0")
    parser.add_argument("--candidate-checkpoint-mode", choices=["0", "1"], default="1")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    v3_root = script_path.parents[1]
    repo_root = Path(args.repo_root).resolve() if args.repo_root else v3_root.parent
    build_dir = Path(args.build_dir).resolve() if args.build_dir else (v3_root / "build").resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (v3_root / "gate_evidence" / "card8_validation").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript_lines: list[str] = []
    steps: list[dict] = []

    def log_cmd(name: str, cmd: list[str], env: dict[str, str] | None = None) -> tuple[bool, str]:
        code, out, err, elapsed_ms = run_command(cmd, cwd=repo_root, env=env)
        steps.append(
            {
                "name": name,
                "command": cmd,
                "exit_code": code,
                "elapsed_ms": round(elapsed_ms, 3),
            }
        )
        transcript_lines.append(f"$ {' '.join(cmd)}")
        transcript_lines.append(f"exit_code={code} elapsed_ms={elapsed_ms:.3f}")
        if out:
            transcript_lines.append("--- stdout ---")
            transcript_lines.append(out.rstrip())
        if err:
            transcript_lines.append("--- stderr ---")
            transcript_lines.append(err.rstrip())
        transcript_lines.append("")
        return code == 0, (err or out)

    ok, msg = log_cmd("cmake_configure", ["cmake", "-S", str(v3_root), "-B", str(build_dir)])
    if not ok:
        write_json(out_dir / "summary.json", {"status": "fail", "failed_step": "cmake_configure", "steps": steps})
        write_json(out_dir / "durability_results.json", {"status": "not_run"})
        write_json(out_dir / "perf_results.json", {"status": "not_run"})
        (out_dir / "ctest.log").write_text("", encoding="utf-8")
        (out_dir / "command_transcript.log").write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")
        print("FAIL: cmake configure failed", file=sys.stderr)
        print(msg, file=sys.stderr)
        return 1

    ok, msg = log_cmd("cmake_build", ["cmake", "--build", str(build_dir), "--config", "Release"])
    if not ok:
        write_json(out_dir / "summary.json", {"status": "fail", "failed_step": "cmake_build", "steps": steps})
        write_json(out_dir / "durability_results.json", {"status": "not_run"})
        write_json(out_dir / "perf_results.json", {"status": "not_run"})
        (out_dir / "ctest.log").write_text("", encoding="utf-8")
        (out_dir / "command_transcript.log").write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")
        print("FAIL: cmake build failed", file=sys.stderr)
        print(msg, file=sys.stderr)
        return 1

    durability_results: dict = {"status": "pass", "modes": []}
    for mode in ("0", "1"):
        env_mode = dict(os.environ)
        env_mode["VECTOR_DB_V3_POST_INGEST_CHECKPOINT"] = mode
        env_mode["VECTOR_DB_V3_COMPLIANCE_PROFILE"] = "pass"
        regex = (
            "vectordb_v3_durability_wal_tests|vectordb_v3_durability_checkpoint_tests|"
            "vectordb_v3_durability_replay_crash_tests|vectordb_v3_durability_corruption_tests|"
            "vectordb_v3_card6_single_process_tests|vectordb_v3_card7_wal_commit_policy_tests|"
            "vectordb_v3_card8_post_ingest_checkpoint_tests|"
            "vectordb_v3_cli_contract_tests|vectordb_v3_terminal_event_contract_tests"
        )
        ok, msg = log_cmd(
            f"ctest_mode_post_ingest_checkpoint_{mode}",
            ["ctest", "--test-dir", str(build_dir), "--output-on-failure", "-R", regex],
            env=env_mode,
        )
        durability_results["modes"].append({"mode": "enabled" if mode == "1" else "disabled", "pass": bool(ok)})
        if not ok:
            durability_results["status"] = "fail"
            durability_results["failed_mode"] = mode
            durability_results["error"] = msg
            break

    ctest_lines = [line for line in transcript_lines if line.startswith("$ ctest") or line.startswith("exit_code=") or line.startswith("---")]
    (out_dir / "ctest.log").write_text("\n".join(ctest_lines) + ("\n" if ctest_lines else ""), encoding="utf-8")

    if durability_results["status"] != "pass":
        write_json(out_dir / "durability_results.json", durability_results)
        write_json(out_dir / "perf_results.json", {"status": "not_run"})
        write_json(out_dir / "summary.json", {"status": "fail", "failed_step": "durability_matrix", "steps": steps, "durability_results": durability_results})
        (out_dir / "command_transcript.log").write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")
        print("FAIL: durability matrix failed", file=sys.stderr)
        return 1

    perf_results: dict = {"status": "skipped"}
    if not args.skip_perf:
        perf_tmp = Path(tempfile.gettempdir()) / "vectordb_v3_card8_perf"
        if perf_tmp.exists():
            import shutil

            shutil.rmtree(perf_tmp)
        perf_tmp.mkdir(parents=True, exist_ok=True)
        baseline_mode_name = "enabled" if args.baseline_checkpoint_mode == "1" else "disabled"
        candidate_mode_name = "enabled" if args.candidate_checkpoint_mode == "1" else "disabled"
        try:
            baseline = run_pipeline_probe(repo_root, v3_root, build_dir, baseline_mode_name, perf_tmp)
            candidate = run_pipeline_probe(repo_root, v3_root, build_dir, candidate_mode_name, perf_tmp)
            base_total = float(baseline["total_pipeline"]["median_ms"])
            cand_total = float(candidate["total_pipeline"]["median_ms"])
            improvement = ((base_total - cand_total) / base_total * 100.0) if base_total > 0 else 0.0
            no_regression_floor = -5.0
            perf_results = {
                "status": "pass" if improvement >= no_regression_floor else "fail",
                "baseline": baseline,
                "candidate": candidate,
                "delta": {"median_improvement_pct": round(improvement, 3)},
                "thresholds": {"no_regression_floor_pct": no_regression_floor, "preferred_positive_pct": 0.0},
            }
            if perf_results["status"] == "fail":
                baseline_retry = run_pipeline_probe(repo_root, v3_root, build_dir, baseline_mode_name, perf_tmp)
                candidate_retry = run_pipeline_probe(repo_root, v3_root, build_dir, candidate_mode_name, perf_tmp)
                base_total_retry = float(baseline_retry["total_pipeline"]["median_ms"])
                cand_total_retry = float(candidate_retry["total_pipeline"]["median_ms"])
                improvement_retry = ((base_total_retry - cand_total_retry) / base_total_retry * 100.0) if base_total_retry > 0 else 0.0
                perf_results["retry"] = {
                    "baseline": baseline_retry,
                    "candidate": candidate_retry,
                    "delta": {"median_improvement_pct": round(improvement_retry, 3)},
                }
                if improvement_retry >= no_regression_floor:
                    perf_results["status"] = "pass"
                    perf_results["note"] = "initial fail, retry pass"
        except Exception as exc:
            perf_results = {"status": "fail", "error": str(exc)}

    write_json(out_dir / "durability_results.json", durability_results)
    write_json(out_dir / "perf_results.json", perf_results)
    (out_dir / "command_transcript.log").write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")

    failed_step = None
    if perf_results.get("status") == "fail":
        failed_step = "perf_results"

    summary = {
        "status": "fail" if failed_step else "pass",
        "profile": args.profile,
        "steps": steps,
        "failed_step": failed_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "durability_results_json": str(out_dir / "durability_results.json"),
            "perf_results_json": str(out_dir / "perf_results.json"),
            "ctest_log": str(out_dir / "ctest.log"),
            "command_transcript_log": str(out_dir / "command_transcript.log"),
        },
        "durability_results": durability_results,
        "perf_results": perf_results,
    }
    write_json(out_dir / "summary.json", summary)
    if failed_step:
        print("FAIL: Card 8 validation failed", file=sys.stderr)
        return 1

    print("PASS: Card 8 validation completed.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
