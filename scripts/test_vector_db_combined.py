from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

STEP_MARKER_RE = re.compile(r"\[step\s+(\d+)\s*/\s*(\d+)\]", re.IGNORECASE)
TIMINGS_FILE = ".vector_db_combined_timings.json"
MAX_TIMING_SAMPLES = 20


def _fmt_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def _progress_bar(elapsed_s: float, estimate_s: float, width: int = 24) -> str:
    if estimate_s <= 0:
        return f"[unknown ETA] elapsed {_fmt_duration(elapsed_s)}"
    ratio = min(0.99, elapsed_s / estimate_s)
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    eta_s = max(0.0, estimate_s - elapsed_s)
    pct = int(ratio * 100.0)
    if elapsed_s > estimate_s:
        overrun_s = elapsed_s - estimate_s
        return (
            f"[{bar}] {pct:3d}% elapsed {_fmt_duration(elapsed_s)} "
            f"overrun +{_fmt_duration(overrun_s)}"
        )
    return f"[{bar}] {pct:3d}% elapsed {_fmt_duration(elapsed_s)} eta {_fmt_duration(eta_s)}"


def _estimate_from_history(default_s: int, samples: list[float]) -> int:
    if not samples:
        return default_s
    ordered = sorted(max(1.0, x) for x in samples)
    n = len(ordered)
    if n < 4:
        learned = sum(ordered) / float(n)
    else:
        idx = min(n - 1, int(0.75 * (n - 1)))
        learned = ordered[idx]
    return max(1, int(round(learned)))


def _load_timings(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, list[float]] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, list):
            continue
        cleaned = [float(v) for v in value if isinstance(v, (int, float)) and float(v) > 0.0]
        if cleaned:
            out[key] = cleaned[-MAX_TIMING_SAMPLES:]
    return out


def _save_timings(path: Path, timings: dict[str, list[float]]) -> None:
    serializable: dict[str, list[float]] = {}
    for key, value in timings.items():
        if not value:
            continue
        serializable[key] = [round(float(v), 3) for v in value[-MAX_TIMING_SAMPLES:]]
    path.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")


def _print_timing_report(estimates_s: dict[str, int], actual_s: dict[str, float]) -> None:
    if not actual_s:
        return
    print("\n== Timing accuracy report ==")
    total_est = 0.0
    total_actual = 0.0
    for name, actual in actual_s.items():
        est = float(estimates_s.get(name, 0))
        total_est += est
        total_actual += actual
        delta = actual - est
        pct_text = f"{(delta / est) * 100.0:+.1f}%" if est > 0 else "n/a"
        print(
            f"- {name}: est {_fmt_duration(est)} vs actual {_fmt_duration(actual)} "
            f"(delta {delta:+.1f}s, {pct_text})"
        )
    total_delta = total_actual - total_est
    total_pct = (total_delta / total_est * 100.0) if total_est > 0 else 0.0
    print(
        f"- TOTAL: est {_fmt_duration(total_est)} vs actual {_fmt_duration(total_actual)} "
        f"(delta {total_delta:+.1f}s, {total_pct:+.1f}%)"
    )


def run_stream_step(name: str, cmd: list[str], cwd: Path, estimate_s: int) -> float:
    print(f"\n== {name} ==")
    print(f"$ {' '.join(cmd)}")
    print(f"Estimated duration: ~{_fmt_duration(estimate_s)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    q: Queue[str | None] = Queue()
    done = threading.Event()

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            q.put(line)
        q.put(None)
        done.set()

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    started = time.monotonic()
    last_tick = 0.0
    status_printed = False
    rc: int | None = None
    substep_idx: int | None = None
    substep_total: int | None = None

    while True:
        drained = False
        while True:
            try:
                item = q.get_nowait()
            except Empty:
                break
            drained = True
            if item is None:
                done.set()
                continue
            if status_printed:
                print()
                status_printed = False
            print(item, end="" if item.endswith("\n") else "\n")
            marker = STEP_MARKER_RE.search(item)
            if marker:
                substep_idx = int(marker.group(1))
                substep_total = int(marker.group(2))

        rc = proc.poll()
        now = time.monotonic()
        if rc is None and now - last_tick >= 1.0:
            elapsed = now - started
            if substep_idx is not None and substep_total is not None and substep_total > 0:
                completed = max(0, min(substep_total, substep_idx - 1))
                if completed > 0:
                    per_step = elapsed / float(completed)
                    total_estimate = elapsed + max(0.0, per_step * float(substep_total - completed))
                else:
                    total_estimate = float(estimate_s)
                print(
                    "\r"
                    + _progress_bar(elapsed, max(total_estimate, 1.0))
                    + f" substeps {completed}/{substep_total}",
                    end="",
                    flush=True,
                )
            else:
                print("\r" + _progress_bar(elapsed, float(estimate_s)), end="", flush=True)
            status_printed = True
            last_tick = now

        if rc is not None and done.is_set() and not drained:
            break
        time.sleep(0.1)

    if status_printed:
        print()
    if rc is None:
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{name} failed with exit code {rc}")
    return time.monotonic() - started


def run_timed(cmd: list[str], cwd: Path) -> tuple[str, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    elapsed_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"elapsed_s: {elapsed_s:.3f}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout.strip(), elapsed_s


def require_keys(obj: dict[str, object], required: list[str], context: str) -> None:
    missing = [k for k in required if k not in obj]
    if missing:
        raise RuntimeError(
            f"{context} is missing expected keys: {missing}. "
            "This usually means the CLI binary is stale; rebuild vectordb_cli and retry."
        )


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def check_dependency(name: str, cmd: list[str], cwd: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    except FileNotFoundError:
        return False, "missing executable"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, tail[-1] if tail else f"exit {proc.returncode}"
    return True, "ok"


def ensure_runtime_artifacts(data_dir: Path, require_cluster: bool) -> None:
    required = [
        data_dir / "manifest.json",
        data_dir / "dirty_ranges.json",
        data_dir / "wal.log",
    ]
    if require_cluster:
        required.append(data_dir / "clusters" / "initial" / "cluster_manifest.json")
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"missing required runtime artifacts: {missing}")


def clear_old_test_paths(vector_db_dir: Path, extra_data_dir: Path, no_clean: bool) -> None:
    stale_paths = [
        vector_db_dir / "smoke_data",
        vector_db_dir / "smoke_data_profile",
        vector_db_dir / "tc_check_data",
        extra_data_dir,
    ]
    print("\n== Cleanup old test paths ==")
    for p in stale_paths:
        if no_clean:
            print(f"- skip cleanup (no-clean): {p}")
            continue
        if p.exists():
            print(f"- removing {p}")
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)
        else:
            print(f"- already clean {p}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combined VectorDB build/test + smoke profile + restart validation."
    )
    parser.add_argument("--skip-configure", action="store_true", help="Skip cmake configure.")
    parser.add_argument("--skip-build", action="store_true", help="Skip cmake build.")
    parser.add_argument("--skip-ctest", action="store_true", help="Skip ctest.")
    parser.add_argument("--skip-generate", action="store_true", help="Skip synthetic dataset generation.")
    parser.add_argument("--payloads", default="synthetic_dataset_10k_fp16/insert_payloads.jsonl", help="Payload JSONL path relative to vector_db unless absolute.")
    parser.add_argument("--data-dir", default="smoke_data_combined", help="Data directory for the smoke run.")
    parser.add_argument("--cluster-seed", default="9001", help="Seed for build-initial-clusters.")
    parser.add_argument("--json-out", default="smoke_cli_combined_report.json", help="Combined report output path relative to repo root unless absolute.")
    parser.add_argument("--keep-data", action="store_true", help="Preserve data directory after success.")
    parser.add_argument("--no-clean", action="store_true", help="Do not remove stale run directories before execution.")
    parser.add_argument("--strict-deps", dest="strict_deps", action="store_true", help="Fail early on dependency preflight issues (default).")
    parser.add_argument("--no-strict-deps", dest="strict_deps", action="store_false", help="Record dependency preflight issues but continue.")
    parser.set_defaults(strict_deps=True)
    parser.add_argument("--count", type=int, default=10_000, help="Synthetic embedding count.")
    parser.add_argument("--dim", type=int, default=1024, help="Synthetic embedding dimension.")
    parser.add_argument("--seed", type=int, default=1337, help="Synthetic generation seed.")
    parser.add_argument("--clusters", type=int, default=32, help="Synthetic latent cluster count.")
    parser.add_argument("--noise-std", type=float, default=0.04, help="Synthetic generation noise std.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization in generation.")
    parser.add_argument("--gpu-generate", action="store_true", help="Attempt CuPy-backed generation.")
    parser.add_argument("--dataset-out", default="vector_db/synthetic_dataset_10k_fp16", help="Synthetic dataset output directory.")
    parser.add_argument(
        "--run-second-level",
        action="store_true",
        help="Run scripts/test_vector_db_second_level.py after combined smoke flow.",
    )
    parser.add_argument(
        "--second-level-report",
        default="vector_db/second_level_test_report.json",
        help="JSON report path passed to second-level validation script.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    vector_db_dir = root / "vector_db"
    build_dir = vector_db_dir / "build"
    bin_name = "vectordb_cli.exe" if sys.platform.startswith("win") else "vectordb_cli"
    cli_path = build_dir / bin_name
    payloads_path = Path(args.payloads)
    if not payloads_path.is_absolute():
        payloads_path = vector_db_dir / payloads_path
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = vector_db_dir / data_dir
    json_out = Path(args.json_out)
    if not json_out.is_absolute():
        json_out = root / json_out
    dataset_out = Path(args.dataset_out)
    if not dataset_out.is_absolute():
        dataset_out = root / dataset_out
    second_level_report = Path(args.second_level_report)
    if not second_level_report.is_absolute():
        second_level_report = root / second_level_report

    print("Vector DB Combined validation (build + smoke profile + restart checks)")
    print("- Includes dependency preflight and explicit restart dependency artifact checks")
    print("- Includes WAL/checkpoint/stats/clustering schema assertions and report output")

    base_estimates_s = {
        "Configure (CMake)": 45,
        "Build": 180,
        "CTest": 90,
        "Generate synthetic dataset": 20,
        "Combined CLI smoke/profile": 240,
        "Second-level validation": 180,
    }
    timing_history = _load_timings(root / TIMINGS_FILE)
    run_actual_s: dict[str, float] = {}
    estimates_s = {
        name: _estimate_from_history(default_s, timing_history.get(name, []))
        for name, default_s in base_estimates_s.items()
    }

    dep_checks: dict[str, dict[str, object]] = {}
    preflight_items: list[tuple[str, bool, str]] = []
    preflight_items.append(("python", *check_dependency("python", [sys.executable, "--version"], root)))
    if not args.skip_configure or not args.skip_build:
        preflight_items.append(("cmake", *check_dependency("cmake", ["cmake", "--version"], root)))
    if not args.skip_ctest:
        preflight_items.append(("ctest", *check_dependency("ctest", ["ctest", "--version"], root)))
    gen_script = root / "scripts" / "generate_synthetic_embeddings.py"
    preflight_items.append(("generator_script", gen_script.exists(), "ok" if gen_script.exists() else "missing script"))
    preflight_items.append(("vector_db_dir", vector_db_dir.exists(), "ok" if vector_db_dir.exists() else "missing folder"))
    if args.skip_build:
        preflight_items.append(("vectordb_cli_binary", cli_path.exists(), "ok" if cli_path.exists() else f"missing binary at {cli_path}"))
    if args.skip_generate:
        preflight_items.append(("payloads", payloads_path.exists(), "ok" if payloads_path.exists() else f"missing payloads at {payloads_path}"))

    print("\n== Dependency preflight ==")
    failed_deps: list[str] = []
    for name, ok, details in preflight_items:
        dep_checks[name] = {"ok": ok, "details": details}
        print(f"- {name}: {'PASS' if ok else 'FAIL'} ({details})")
        if not ok:
            failed_deps.append(name)

    if failed_deps and args.strict_deps:
        print(f"[FAIL] dependency preflight failed: {failed_deps}")
        return 1

    steps: list[dict[str, object]] = []

    def do(label: str, cmd: list[str]) -> str:
        print(f"[profile step {len(steps) + 1}] {label}")
        stdout, elapsed_s = run_timed(cmd, vector_db_dir)
        print(f"  elapsed: {elapsed_s:.3f}s")
        steps.append(
            {
                "step": len(steps) + 1,
                "label": label,
                "command": cmd,
                "elapsed_s": round(elapsed_s, 6),
            }
        )
        return stdout

    try:
        clear_old_test_paths(vector_db_dir, data_dir, args.no_clean)

        if not args.skip_configure:
            elapsed = run_stream_step(
                "Configure (CMake)",
                ["cmake", "-S", str(vector_db_dir), "-B", str(build_dir)],
                cwd=root,
                estimate_s=estimates_s["Configure (CMake)"],
            )
            run_actual_s["Configure (CMake)"] = elapsed
            timing_history.setdefault("Configure (CMake)", []).append(elapsed)

        if not args.skip_build:
            elapsed = run_stream_step(
                "Build",
                ["cmake", "--build", str(build_dir), "--config", "Release"],
                cwd=root,
                estimate_s=estimates_s["Build"],
            )
            run_actual_s["Build"] = elapsed
            timing_history.setdefault("Build", []).append(elapsed)

        if not cli_path.exists():
            raise RuntimeError(f"missing CLI binary at {cli_path}; build first with cmake")

        if not args.skip_ctest:
            elapsed = run_stream_step(
                "CTest",
                ["ctest", "--test-dir", str(build_dir), "--output-on-failure"],
                cwd=root,
                estimate_s=estimates_s["CTest"],
            )
            run_actual_s["CTest"] = elapsed
            timing_history.setdefault("CTest", []).append(elapsed)

        if not args.skip_generate:
            gen_cmd = [
                sys.executable,
                "scripts/generate_synthetic_embeddings.py",
                "--count",
                str(args.count),
                "--dim",
                str(args.dim),
                "--seed",
                str(args.seed),
                "--clusters",
                str(args.clusters),
                "--noise-std",
                str(args.noise_std),
                "--out-dir",
                str(dataset_out),
            ]
            if args.no_normalize:
                gen_cmd.append("--no-normalize")
            if args.gpu_generate:
                gen_cmd.append("--gpu")
            elapsed = run_stream_step(
                "Generate synthetic dataset",
                gen_cmd,
                cwd=root,
                estimate_s=estimates_s["Generate synthetic dataset"],
            )
            run_actual_s["Generate synthetic dataset"] = elapsed
            timing_history.setdefault("Generate synthetic dataset", []).append(elapsed)

        if not payloads_path.exists():
            raise RuntimeError(f"missing payloads file: {payloads_path}")
        payload_count = count_jsonl_rows(payloads_path)
        if payload_count < 12:
            raise RuntimeError("payload file has too few rows; expected at least 12")

        print("\n== Combined CLI smoke/profile ==")
        smoke_started = time.perf_counter()

        do("init store", [str(cli_path), "init", "--path", str(data_dir)])
        do("bulk insert payloads", [str(cli_path), "bulk-insert", "--path", str(data_dir), "--input", str(payloads_path)])
        do("read row 100", [str(cli_path), "get", "--path", str(data_dir), "--id", "100"])
        do("update row 100 meta", [str(cli_path), "update-meta", "--path", str(data_dir), "--id", "100", "--meta", '{"kind":"b","x":1}'])
        do("delete row 100", [str(cli_path), "delete", "--path", str(data_dir), "--id", "100"])

        wal_before_checkpoint = json.loads(do("check WAL before checkpoint", [str(cli_path), "wal-stats", "--path", str(data_dir)]))
        assert wal_before_checkpoint["wal_entries"] >= 3

        stats_out = do("collect stats", [str(cli_path), "stats", "--path", str(data_dir)])
        parsed = json.loads(stats_out)
        assert parsed["total_rows"] == payload_count
        assert parsed["tombstone_rows"] == 1
        ensure_runtime_artifacts(data_dir, require_cluster=False)

        do("checkpoint", [str(cli_path), "checkpoint", "--path", str(data_dir)])
        wal_after_checkpoint = json.loads(do("check WAL after checkpoint", [str(cli_path), "wal-stats", "--path", str(data_dir)]))
        assert wal_after_checkpoint["wal_entries"] == 0
        assert wal_after_checkpoint["checkpoint_lsn"] >= wal_before_checkpoint["last_lsn"]

        do("build initial clusters", [str(cli_path), "build-initial-clusters", "--path", str(data_dir), "--seed", str(args.cluster_seed)])
        cluster_stats = json.loads(do("read cluster stats", [str(cli_path), "cluster-stats", "--path", str(data_dir)]))
        cluster_health = json.loads(do("read cluster health", [str(cli_path), "cluster-health", "--path", str(data_dir)]))
        assert cluster_stats["available"] is True
        assert cluster_stats["chosen_k"] >= cluster_stats["k_min"]
        assert cluster_stats["chosen_k"] <= cluster_stats["k_max"]
        require_keys(
            cluster_stats,
            [
                "gpu_backend",
                "tensor_core_enabled",
                "scoring_ms_total",
                "scoring_calls",
                "elbow_k_evaluated_count",
                "elbow_stage_a_candidates",
                "elbow_stage_b_candidates",
                "elbow_early_stop_reason",
                "stability_runs_executed",
                "load_live_vectors_ms",
                "id_estimation_ms",
                "elbow_ms",
                "stability_ms",
                "write_artifacts_ms",
                "total_build_ms",
                "live_vector_bytes_read",
                "live_vector_contiguous_spans",
                "live_vector_sparse_reads",
                "live_vector_sparse_fallback",
                "live_vector_async_double_buffer",
                "elbow_stage_a_approx_enabled",
                "elbow_stage_a_approx_dim",
                "elbow_stage_a_approx_stride",
                "elbow_stage_b_pruned_candidates",
                "elbow_stage_b_window_k_min",
                "elbow_stage_b_window_k_max",
                "elbow_stage_b_prune_reason",
                "elbow_int8_search_enabled",
                "elbow_int8_tensor_core_used",
                "elbow_int8_eval_count",
                "elbow_int8_scale_mode",
                "elbow_scoring_precision",
            ],
            "cluster-stats output",
        )
        assert cluster_stats["elbow_stage_b_candidates"] >= 2
        assert cluster_stats["elbow_stage_b_pruned_candidates"] <= cluster_stats["elbow_stage_b_candidates"]
        assert cluster_stats["elbow_stage_b_window_k_min"] >= cluster_stats["k_min"]
        assert cluster_stats["elbow_stage_b_window_k_max"] <= cluster_stats["k_max"]
        assert cluster_stats["elbow_int8_eval_count"] >= 0
        assert cluster_stats["elbow_scoring_precision"] in ("fp16", "int8-search/fp16-final")
        assert cluster_stats["elbow_k_evaluated_count"] >= cluster_stats["elbow_stage_b_candidates"]
        assert cluster_health["available"] is True

        ensure_runtime_artifacts(data_dir, require_cluster=True)
        get_after_restart = json.loads(
            do("restart check get row 100", [str(cli_path), "get", "--path", str(data_dir), "--id", "100"])
        )
        assert get_after_restart["deleted"] is True
        stats_after_restart = json.loads(do("restart check stats", [str(cli_path), "stats", "--path", str(data_dir)]))
        assert stats_after_restart["total_rows"] == payload_count
        assert stats_after_restart["tombstone_rows"] == 1
        assert stats_after_restart["dirty_ranges"] >= 3
        cluster_stats_after = json.loads(do("restart check cluster stats", [str(cli_path), "cluster-stats", "--path", str(data_dir)]))
        assert cluster_stats_after["version"] == cluster_stats["version"]
        cluster_stats_after2 = json.loads(do("restart check cluster stats repeated", [str(cli_path), "cluster-stats", "--path", str(data_dir)]))
        assert cluster_stats_after2["version"] == cluster_stats_after["version"]
        ensure_runtime_artifacts(data_dir, require_cluster=True)

        smoke_elapsed = time.perf_counter() - smoke_started
        run_actual_s["Combined CLI smoke/profile"] = smoke_elapsed
        timing_history.setdefault("Combined CLI smoke/profile", []).append(smoke_elapsed)

        if args.run_second_level:
            second_level_cmd = [
                sys.executable,
                "scripts/test_vector_db_second_level.py",
                "--seed",
                str(args.cluster_seed),
                "--json-out",
                str(second_level_report),
            ]
            if args.keep_data:
                second_level_cmd.append("--keep-data")
            elapsed = run_stream_step(
                "Second-level validation",
                second_level_cmd,
                cwd=root,
                estimate_s=180,
            )
            run_actual_s["Second-level validation"] = elapsed
            timing_history.setdefault("Second-level validation", []).append(elapsed)
    except FileNotFoundError as e:
        print(f"[FAIL] Missing tool: {e}")
        return 1
    except (RuntimeError, AssertionError, json.JSONDecodeError) as e:
        print(f"[FAIL] {e}")
        return 1
    finally:
        _print_timing_report(estimates_s, run_actual_s)
        try:
            _save_timings(root / TIMINGS_FILE, timing_history)
        except OSError:
            pass

    total_s = sum(float(s["elapsed_s"]) for s in steps)
    ranked = sorted(steps, key=lambda x: float(x["elapsed_s"]), reverse=True)

    print("\n=== Combined Smoke Profile Report ===")
    print(f"Total measured command time: {total_s:.3f}s")
    for s in ranked:
        pct = (float(s["elapsed_s"]) / total_s * 100.0) if total_s > 0 else 0.0
        print(f"- #{s['step']:>2} {s['label']}: {float(s['elapsed_s']):.3f}s ({pct:.1f}%)")

    report = {
        "total_elapsed_s": round(total_s, 6),
        "steps_ranked": ranked,
        "steps_in_order": steps,
        "dependency_preflight": dep_checks,
        "meta": {
            "data_dir": str(data_dir),
            "payloads_path": str(payloads_path),
            "cluster_seed": str(args.cluster_seed),
            "skip_configure": args.skip_configure,
            "skip_build": args.skip_build,
            "skip_ctest": args.skip_ctest,
            "skip_generate": args.skip_generate,
            "payload_count": count_jsonl_rows(payloads_path),
            "run_second_level": args.run_second_level,
            "second_level_report": str(second_level_report),
        },
    }
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote JSON report: {json_out}")

    if not args.keep_data and data_dir.exists():
        shutil.rmtree(data_dir)

    print("\n[PASS] Combined Vector DB smoke/profile checks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
