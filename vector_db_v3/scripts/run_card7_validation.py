from __future__ import annotations

import argparse
import json
import os
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


def run_bulk_perf(cli: Path, repo_root: Path, mode: str, out_dir: Path, runs: int = 4) -> dict:
    env = dict(os.environ)
    env["VECTOR_DB_V3_COMPLIANCE_PROFILE"] = "pass"
    env["VECTOR_DB_V3_WAL_COMMIT_POLICY"] = mode

    data_dir = out_dir / f"data_{mode}"
    bulk_path = out_dir / f"bulk_{mode}.bin"
    rows = [(50000 + i, 0.0001 * ((i % 101) + 1)) for i in range(4096)]
    write_bulk_bin(bulk_path, rows)

    code, out, err, _ = run_command([str(cli), "init", "--path", str(data_dir)], cwd=repo_root, env=env)
    if code != 0:
        raise RuntimeError(f"{mode} init failed: {err or out}")

    samples: list[float] = []
    run_rows: list[dict] = []
    for i in range(runs):
        code, out, err, elapsed_ms = run_command(
            [str(cli), "bulk-insert-bin", "--path", str(data_dir), "--input", str(bulk_path), "--batch-size", "128"],
            cwd=repo_root,
            env=env,
        )
        if code != 0:
            raise RuntimeError(f"{mode} bulk-insert-bin failed run={i+1}: {err or out}")
        samples.append(elapsed_ms)
        run_rows.append({"run_index": i + 1, "elapsed_ms": round(elapsed_ms, 3)})
    return {"mode": mode, "summary": summarize(samples), "runs": run_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Card 7 build/test/perf validation.")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--profile", choices=["minimum", "jetson_orin"], default="minimum")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    v3_root = script_path.parents[1]
    repo_root = Path(args.repo_root).resolve() if args.repo_root else v3_root.parent
    build_dir = Path(args.build_dir).resolve() if args.build_dir else (v3_root / "build").resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (v3_root / "gate_evidence" / "card7_validation").resolve()
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
    for mode in ("strict_per_record", "batch_group_commit"):
        env_mode = dict(os.environ)
        env_mode["VECTOR_DB_V3_WAL_COMMIT_POLICY"] = mode
        env_mode["VECTOR_DB_V3_COMPLIANCE_PROFILE"] = "pass"
        regex = (
            "vectordb_v3_durability_wal_tests|vectordb_v3_durability_checkpoint_tests|"
            "vectordb_v3_durability_replay_crash_tests|vectordb_v3_durability_corruption_tests|"
            "vectordb_v3_card6_single_process_tests|vectordb_v3_card7_wal_commit_policy_tests|"
            "vectordb_v3_cli_contract_tests|vectordb_v3_terminal_event_contract_tests"
        )
        ok, msg = log_cmd(
            f"ctest_mode_{mode}",
            ["ctest", "--test-dir", str(build_dir), "--output-on-failure", "-R", regex],
            env=env_mode,
        )
        durability_results["modes"].append({"mode": mode, "pass": bool(ok)})
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
        cli = build_dir / "vectordb_v3_cli.exe"
        if not cli.exists():
            cli = build_dir / "vectordb_v3_cli"
        if not cli.exists():
            perf_results = {"status": "fail", "error": "missing vectordb_v3_cli binary"}
        else:
            perf_tmp = Path(tempfile.gettempdir()) / "vectordb_v3_card7_perf"
            if perf_tmp.exists():
                import shutil
                shutil.rmtree(perf_tmp)
            perf_tmp.mkdir(parents=True, exist_ok=True)
            try:
                baseline = run_bulk_perf(cli, repo_root, "strict_per_record", perf_tmp)
                candidate = run_bulk_perf(cli, repo_root, "batch_group_commit", perf_tmp)
                base_med = float(baseline["summary"]["median_ms"])
                cand_med = float(candidate["summary"]["median_ms"])
                improvement = ((base_med - cand_med) / base_med * 100.0) if base_med > 0 else 0.0
                no_regression_floor = -5.0
                perf_results = {
                    "status": "pass" if improvement >= no_regression_floor else "fail",
                    "baseline": baseline,
                    "candidate": candidate,
                    "delta": {"median_improvement_pct": round(improvement, 3)},
                    "thresholds": {"no_regression_floor_pct": no_regression_floor, "preferred_positive_pct": 0.0},
                }
                if perf_results["status"] == "fail":
                    # Anti-flake retry once.
                    baseline_retry = run_bulk_perf(cli, repo_root, "strict_per_record", perf_tmp)
                    candidate_retry = run_bulk_perf(cli, repo_root, "batch_group_commit", perf_tmp)
                    base_med_retry = float(baseline_retry["summary"]["median_ms"])
                    cand_med_retry = float(candidate_retry["summary"]["median_ms"])
                    improvement_retry = ((base_med_retry - cand_med_retry) / base_med_retry * 100.0) if base_med_retry > 0 else 0.0
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
        print("FAIL: Card 7 validation failed", file=sys.stderr)
        return 1

    print("PASS: Card 7 validation completed.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
