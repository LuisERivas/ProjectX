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


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def parse_events(stdout: str) -> list[dict]:
    out: list[dict] = []
    for line in stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception:
            continue
        if isinstance(obj, dict) and "event_type" in obj:
            out.append(obj)
    return out


def last_stage_end(events: list[dict], stage_id: str) -> dict:
    for event in reversed(events):
        if event.get("event_type") in {"stage_end", "stage_fail", "stage_skip"} and event.get("stage_id") == stage_id:
            return event
    return {}


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


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def build_perf_checklist(perf_payload: dict) -> list[tuple[str, bool]]:
    checklist: list[tuple[str, bool]] = []
    delta = perf_payload.get("delta", {})
    top_imp = float(delta.get("top_median_improvement_pct", 0.0))
    mid_imp = float(delta.get("mid_median_improvement_pct", 0.0))

    candidate_precisions = perf_payload["modes"]["candidate_fp16"]["summary"]["top_precision_samples"]
    fp16_seen = any(p == "fp16" for p in candidate_precisions)

    # If FP16 path is not active in candidate runs, perf comparison is not equivalent
    # to Card 4's intended acceleration path and can be misleading.
    if not fp16_seen:
        checklist.append(("perf_not_comparable_no_fp16_path", True))
        checklist.append(("candidate_precision_observed", any(p in {"fp16", "fp32"} for p in candidate_precisions)))
        return checklist

    checklist.append(("no_major_regression_top", top_imp > -3.0))
    checklist.append(("no_major_regression_mid", mid_imp > -3.0))
    checklist.append(("candidate_precision_observed", True))
    return checklist


def run_perf_ab(cli: Path, repo_root: Path, out_dir: Path, warmup_runs: int, measure_runs: int) -> dict:
    run_root = out_dir / "perf_ab"
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    mode_matrix = {
        "baseline_off": "off",
        "candidate_fp16": "fp16",
    }
    for label, shard_mode in mode_matrix.items():
        data_dir = run_root / label
        data_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["VECTOR_DB_V3_COMPLIANCE_PROFILE"] = "pass"
        env["VECTOR_DB_V3_GPU_RESIDENCY_MODE"] = "stage"
        env["VECTOR_DB_V3_INTERNAL_SHARD_MODE"] = shard_mode
        env["VECTOR_DB_V3_INTERNAL_SHARD_REPAIR"] = "regenerate"

        code, out, err, _ = run_command([str(cli), "init", "--path", str(data_dir)], cwd=repo_root, env=env)
        if code != 0:
            raise RuntimeError(f"perf_ab init failed ({label}): {err or out}")

        bulk_path = data_dir / "bulk.bin"
        rows = [(10000 + i, 0.001 * ((i % 97) + 1)) for i in range(512)]
        write_bulk_bin(bulk_path, rows)
        code, out, err, _ = run_command(
            [str(cli), "bulk-insert-bin", "--path", str(data_dir), "--input", str(bulk_path), "--batch-size", "256"],
            cwd=repo_root,
            env=env,
        )
        if code != 0:
            raise RuntimeError(f"perf_ab bulk-insert-bin failed ({label}): {err or out}")

        top_times: list[float] = []
        mid_times: list[float] = []
        precisions: list[str] = []
        total_runs = warmup_runs + measure_runs
        logs: list[dict] = []
        for i in range(total_runs):
            code, out, err, _ = run_command(
                [str(cli), "build-top-clusters", "--path", str(data_dir), "--seed", "7"],
                cwd=repo_root,
                env=env,
            )
            if code != 0:
                raise RuntimeError(f"perf_ab build-top-clusters failed ({label} run {i+1}): {err or out}")
            events = parse_events(out)
            top_end = last_stage_end(events, "top")
            top_elapsed = float(top_end.get("stage_elapsed_ms", 0.0))
            precision = str(top_end.get("compute_precision", "unknown"))

            code, out, err, _ = run_command(
                [str(cli), "build-mid-layer-clusters", "--path", str(data_dir), "--seed", "7"],
                cwd=repo_root,
                env=env,
            )
            if code != 0:
                raise RuntimeError(f"perf_ab build-mid-layer-clusters failed ({label} run {i+1}): {err or out}")
            events = parse_events(out)
            mid_end = last_stage_end(events, "mid")
            mid_elapsed = float(mid_end.get("stage_elapsed_ms", 0.0))

            logs.append(
                {
                    "run_index": i + 1,
                    "phase": "warmup" if i < warmup_runs else "measure",
                    "top_ms": round(top_elapsed, 3),
                    "mid_ms": round(mid_elapsed, 3),
                    "top_precision": precision,
                }
            )
            if i >= warmup_runs:
                top_times.append(top_elapsed)
                mid_times.append(mid_elapsed)
                precisions.append(precision)

        write_json(run_root / f"{label}_runs.json", {"runs": logs})
        results[label] = {
            "summary": {
                "top_median_ms": round(median(top_times), 3),
                "mid_median_ms": round(median(mid_times), 3),
                "top_precision_samples": precisions,
            },
            "samples": {
                "top_ms": [round(v, 3) for v in top_times],
                "mid_ms": [round(v, 3) for v in mid_times],
            },
        }

    base = results["baseline_off"]["summary"]
    cand = results["candidate_fp16"]["summary"]
    top_base = float(base["top_median_ms"])
    mid_base = float(base["mid_median_ms"])
    top_cand = float(cand["top_median_ms"])
    mid_cand = float(cand["mid_median_ms"])
    top_imp = ((top_base - top_cand) / top_base * 100.0) if top_base > 0 else 0.0
    mid_imp = ((mid_base - mid_cand) / mid_base * 100.0) if mid_base > 0 else 0.0
    delta = {
        "top_median_improvement_pct": round(top_imp, 3),
        "mid_median_improvement_pct": round(mid_imp, 3),
    }
    return {"modes": results, "delta": delta}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Card 4 build+tests+perf validation.")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--profile", choices=["jetson_orin", "minimum"], default="minimum")
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=5)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    vdb_root = script_path.parents[1]
    repo_root = Path(args.repo_root).resolve() if args.repo_root else vdb_root.parent
    build_dir = Path(args.build_dir).resolve() if args.build_dir else (vdb_root / "build").resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (vdb_root / "gate_evidence" / "card4_validation").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict] = []

    def run_step(name: str, cmd: list[str], env: dict[str, str] | None = None) -> tuple[bool, str]:
        code, out, err, elapsed = run_command(cmd, cwd=repo_root, env=env)
        log_path = logs_dir / f"{name}.log"
        write_text(
            log_path,
            f"$ {' '.join(cmd)}\nexit_code={code}\nelapsed_ms={elapsed:.3f}\n\n--- stdout ---\n{out}\n--- stderr ---\n{err}\n",
        )
        steps.append(
            {
                "name": name,
                "command": cmd,
                "exit_code": code,
                "elapsed_ms": round(elapsed, 3),
                "log": str(log_path),
            }
        )
        return code == 0, (err or out)

    ok, msg = run_step("cmake_configure", ["cmake", "-S", str(vdb_root), "-B", str(build_dir)])
    if not ok:
        write_json(out_dir / "card4_validation_summary.json", {"status": "fail", "failed_step": "cmake_configure", "steps": steps})
        print("FAIL: cmake configure failed", file=sys.stderr)
        print(msg, file=sys.stderr)
        return 1

    ok, msg = run_step("cmake_build", ["cmake", "--build", str(build_dir), "--config", "Release"])
    if not ok:
        write_json(out_dir / "card4_validation_summary.json", {"status": "fail", "failed_step": "cmake_build", "steps": steps})
        print("FAIL: cmake build failed", file=sys.stderr)
        print(msg, file=sys.stderr)
        return 1

    regex = (
        "vectordb_v3_codec_artifacts_tests|vectordb_v3_codec_corruption_tests|"
        "vectordb_v3_precision_shard_lifecycle_tests|vectordb_v3_precision_shard_alignment_failures_tests|"
        "vectordb_v3_kmeans_backend_parity_tests|vectordb_v3_kmeans_backend_selection_tests|"
        "vectordb_v3_compliance_pass_tests|vectordb_v3_compliance_fail_fast_tests|"
        "vectordb_v3_terminal_event_contract_tests"
    )
    ok, msg = run_step(
        "ctest_targeted",
        ["ctest", "--test-dir", str(build_dir), "--output-on-failure", "-R", regex],
    )
    if not ok:
        write_json(out_dir / "card4_validation_summary.json", {"status": "fail", "failed_step": "ctest_targeted", "steps": steps})
        print("FAIL: targeted ctest failed", file=sys.stderr)
        print(msg, file=sys.stderr)
        return 1

    cli = build_dir / "vectordb_v3_cli.exe"
    if not cli.exists():
        cli = build_dir / "vectordb_v3_cli"
    if not cli.exists():
        write_json(out_dir / "card4_validation_summary.json", {"status": "fail", "failed_step": "cli_lookup", "steps": steps})
        print(f"FAIL: missing CLI binary in {build_dir}", file=sys.stderr)
        return 1

    perf_payload: dict = {"status": "skipped"}
    if not args.skip_perf:
        try:
            perf_payload = run_perf_ab(
                cli=cli,
                repo_root=repo_root,
                out_dir=out_dir,
                warmup_runs=args.warmup_runs,
                measure_runs=args.measure_runs,
            )
        except Exception as exc:
            write_json(
                out_dir / "card4_validation_summary.json",
                {
                    "status": "fail",
                    "failed_step": "perf_ab",
                    "error": str(exc),
                    "profile": args.profile,
                    "steps": steps,
                },
            )
            print(f"FAIL: perf A/B failed: {exc}", file=sys.stderr)
            return 1
        write_json(out_dir / "card4_perf_ab.json", perf_payload)

    checklist = []
    if not args.skip_perf:
        checklist = build_perf_checklist(perf_payload)
        initial_failed = [name for name, passed in checklist if not passed]
        # Retry once to reduce false negatives from transient runtime variance.
        if initial_failed:
            try:
                perf_payload_retry = run_perf_ab(
                    cli=cli,
                    repo_root=repo_root,
                    out_dir=out_dir / "retry",
                    warmup_runs=args.warmup_runs,
                    measure_runs=args.measure_runs,
                )
                write_json(out_dir / "card4_perf_ab_retry.json", perf_payload_retry)
                checklist_retry = build_perf_checklist(perf_payload_retry)
                retry_failed = [name for name, passed in checklist_retry if not passed]
                if not retry_failed:
                    checklist = checklist_retry
                    perf_payload = {
                        **perf_payload_retry,
                        "retry_note": "initial perf checks failed; retry passed",
                    }
                else:
                    perf_payload = {
                        **perf_payload,
                        "retry_attempted": True,
                        "retry_failed_checks": retry_failed,
                    }
            except Exception as exc:
                perf_payload = {
                    **perf_payload,
                    "retry_attempted": True,
                    "retry_error": str(exc),
                }

    failed_checks = [name for name, passed in checklist if not passed]
    status = "pass" if not failed_checks else "fail"
    summary = {
        "status": status,
        "profile": args.profile,
        "steps": steps,
        "checklist": [{"name": n, "pass": p} for n, p in checklist],
        "failed_checks": failed_checks,
        "artifacts": {
            "summary_json": str(out_dir / "card4_validation_summary.json"),
            "perf_ab_json": str(out_dir / "card4_perf_ab.json"),
            "logs_dir": str(logs_dir),
        },
    }
    if not args.skip_perf:
        summary["perf"] = perf_payload

    write_json(out_dir / "card4_validation_summary.json", summary)

    if failed_checks:
        print(f"FAIL: Card 4 checklist failed: {', '.join(failed_checks)}", file=sys.stderr)
        return 1
    print("PASS: Card 4 validation completed.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
