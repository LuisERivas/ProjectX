from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
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


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def summarize(samples: list[float]) -> dict:
    if not samples:
        return {"samples_ms": [], "median_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(samples)
    median = statistics.median(ordered)
    p95 = ordered[max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))]
    return {
        "samples_ms": [round(v, 3) for v in samples],
        "median_ms": round(median, 3),
        "p95_ms": round(p95, 3),
    }


def run_pipeline_probe(
    repo_root: Path,
    v3_root: Path,
    build_dir: Path,
    mode: str,
    runs: int = 3,
) -> dict:
    samples: list[float] = []
    log_rows: list[dict] = []
    for i in range(runs):
        out_path = v3_root / "gate_evidence" / "card6_validation" / f"pipeline_{mode}_{i+1}.json"
        cmd = [
            sys.executable,
            str(v3_root / "scripts" / "pipeline_test.py"),
            "--build-dir",
            str(build_dir),
            "--embedding-count",
            "2048",
            "--input-format",
            "bin",
            "--run-full-pipeline",
            "--orchestration-mode",
            mode,
            "--results-out",
            str(out_path),
        ]
        code, out, err, elapsed_ms = run_command(cmd, cwd=repo_root)
        log_rows.append(
            {
                "run_index": i + 1,
                "exit_code": code,
                "elapsed_ms": round(elapsed_ms, 3),
                "stderr": err.strip()[:2000],
            }
        )
        if code != 0:
            raise RuntimeError(f"pipeline probe failed mode={mode} run={i+1}: {(err or out)[:300]}")
        samples.append(elapsed_ms)
    return {"mode": mode, "summary": summarize(samples), "runs": log_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Card 6 build/test/parity/perf validation.")
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
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (v3_root / "gate_evidence" / "card6_validation").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict] = []

    def run_step(name: str, cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> tuple[bool, str]:
        code, out, err, elapsed_ms = run_command(cmd, cwd=repo_root, env=env)
        write_text(
            log_path,
            f"$ {' '.join(cmd)}\nexit_code={code}\nelapsed_ms={elapsed_ms:.3f}\n\n--- stdout ---\n{out}\n--- stderr ---\n{err}\n",
        )
        steps.append(
            {
                "name": name,
                "command": cmd,
                "exit_code": code,
                "elapsed_ms": round(elapsed_ms, 3),
                "log": str(log_path),
            }
        )
        return code == 0, (err or out)

    ok, msg = run_step("cmake_configure", ["cmake", "-S", str(v3_root), "-B", str(build_dir)], out_dir / "cmake_configure.log")
    if not ok:
        write_json(out_dir / "summary.json", {"status": "fail", "failed_step": "cmake_configure", "steps": steps})
        print("FAIL: cmake configure failed", file=sys.stderr)
        print(msg, file=sys.stderr)
        return 1

    ok, msg = run_step("cmake_build", ["cmake", "--build", str(build_dir), "--config", "Release"], out_dir / "cmake_build.log")
    if not ok:
        write_json(out_dir / "summary.json", {"status": "fail", "failed_step": "cmake_build", "steps": steps})
        print("FAIL: cmake build failed", file=sys.stderr)
        print(msg, file=sys.stderr)
        return 1

    ctest_regex = (
        "vectordb_v3_cli_contract_tests|vectordb_v3_terminal_event_contract_tests|"
        "vectordb_v3_durability_wal_tests|vectordb_v3_durability_checkpoint_tests|"
        "vectordb_v3_durability_replay_crash_tests|vectordb_v3_durability_corruption_tests|"
        "vectordb_v3_card6_single_process_tests"
    )
    ok, msg = run_step(
        "ctest_targeted",
        ["ctest", "--test-dir", str(build_dir), "--output-on-failure", "-R", ctest_regex],
        out_dir / "ctest.log",
    )
    if not ok:
        write_json(out_dir / "summary.json", {"status": "fail", "failed_step": "ctest_targeted", "steps": steps})
        print("FAIL: targeted ctest failed", file=sys.stderr)
        print(msg, file=sys.stderr)
        return 1

    contract_log = out_dir / "contract_tests.log"
    env_pass = dict(os.environ)
    env_pass["VECTOR_DB_V3_COMPLIANCE_PROFILE"] = "pass"
    contract_cmds = [
        [sys.executable, str(v3_root / "scripts" / "test_cli_contract.py")],
        [sys.executable, str(v3_root / "scripts" / "test_terminal_event_contract.py")],
    ]
    contract_lines: list[str] = []
    for cmd in contract_cmds:
        code, out, err, elapsed_ms = run_command(cmd, cwd=repo_root, env=env_pass)
        contract_lines.append(f"$ {' '.join(cmd)}")
        contract_lines.append(f"exit_code={code} elapsed_ms={elapsed_ms:.3f}")
        if out:
            contract_lines.append("--- stdout ---")
            contract_lines.append(out)
        if err:
            contract_lines.append("--- stderr ---")
            contract_lines.append(err)
        contract_lines.append("")
        if code != 0:
            write_text(contract_log, "\n".join(contract_lines))
            write_json(out_dir / "summary.json", {"status": "fail", "failed_step": "contract_tests", "steps": steps})
            print("FAIL: contract tests failed", file=sys.stderr)
            return 1
    write_text(contract_log, "\n".join(contract_lines))

    parity_report = out_dir / "parity_report.json"
    parity_cmd = [
        sys.executable,
        str(v3_root / "scripts" / "test_card6_single_process_pipeline.py"),
        "--build-dir",
        str(build_dir),
        "--report-out",
        str(parity_report),
    ]
    ok, msg = run_step("card6_parity", parity_cmd, out_dir / "parity.log", env=env_pass)
    if not ok:
        write_json(out_dir / "summary.json", {"status": "fail", "failed_step": "card6_parity", "steps": steps})
        print("FAIL: card6 parity failed", file=sys.stderr)
        print(msg, file=sys.stderr)
        return 1

    perf_report: dict = {"status": "skipped"}
    if not args.skip_perf:
        try:
            baseline = run_pipeline_probe(repo_root, v3_root, build_dir, "legacy")
            candidate = run_pipeline_probe(repo_root, v3_root, build_dir, "composite")
            base_med = float(baseline["summary"]["median_ms"])
            cand_med = float(candidate["summary"]["median_ms"])
            improvement = ((base_med - cand_med) / base_med * 100.0) if base_med > 0 else 0.0
            perf_report = {
                "status": "pass" if improvement >= -5.0 else "fail",
                "baseline": baseline,
                "candidate": candidate,
                "delta": {"median_improvement_pct": round(improvement, 3)},
                "thresholds": {"no_regression_floor_pct": -5.0},
            }
            write_json(out_dir / "perf_report.json", perf_report)
            if perf_report["status"] != "pass":
                write_json(out_dir / "summary.json", {"status": "fail", "failed_step": "perf_report", "steps": steps, "perf_report": perf_report})
                print("FAIL: perf regression exceeded allowed floor", file=sys.stderr)
                return 1
        except Exception as exc:
            write_json(out_dir / "summary.json", {"status": "fail", "failed_step": "perf_probe", "steps": steps, "error": str(exc)})
            print(f"FAIL: perf probe failed: {exc}", file=sys.stderr)
            return 1

    summary = {
        "status": "pass",
        "profile": args.profile,
        "steps": steps,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "ctest_log": str(out_dir / "ctest.log"),
            "contract_tests_log": str(contract_log),
            "parity_report": str(parity_report),
            "perf_report": str(out_dir / "perf_report.json") if not args.skip_perf else None,
        },
        "perf": perf_report,
    }
    write_json(out_dir / "summary.json", summary)
    print("PASS: Card 6 validation completed.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
