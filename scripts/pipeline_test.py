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
TIMINGS_FILE = ".vector_db_pipeline_test_timings.json"
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


def run_stream_step(name: str, cmd: list[str], cwd: Path, estimate_s: int) -> tuple[str, float]:
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
    lines: list[str] = []

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
            lines.append(item)
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
        raise RuntimeError(
            f"{name} failed with exit code {rc}\n"
            f"output:\n{''.join(lines)}"
        )
    return "".join(lines).strip(), time.monotonic() - started


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def check_dependency(cmd: list[str], cwd: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    except FileNotFoundError:
        return False, "missing executable"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, tail[-1] if tail else f"exit {proc.returncode}"
    return True, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild vector_db and run full first+second layer clustering pipeline."
    )
    parser.add_argument("--skip-configure", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-ctest", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--data-dir", default="smoke_data_pipeline")
    parser.add_argument("--payloads", default="synthetic_dataset_10k_fp16/insert_payloads.jsonl")
    parser.add_argument("--seed", default="9001")
    parser.add_argument("--source-version", default="")
    parser.add_argument("--json-out", default="vector_db/pipeline_test_report.json")
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--strict-deps", dest="strict_deps", action="store_true")
    parser.add_argument("--no-strict-deps", dest="strict_deps", action="store_false")
    parser.set_defaults(strict_deps=True)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--gen-seed", type=int, default=1337)
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--noise-std", type=float, default=0.04)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--gpu-generate", action="store_true")
    parser.add_argument("--dataset-out", default="vector_db/synthetic_dataset_10k_fp16")
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

    print("Vector DB Pipeline Test (rebuild + first-layer + second-layer)")

    base_estimates_s = {
        "Configure (CMake)": 45,
        "Build": 180,
        "CTest": 90,
        "Generate synthetic dataset": 20,
        "Pipeline CLI flow": 240,
    }
    timing_history = _load_timings(root / TIMINGS_FILE)
    run_actual_s: dict[str, float] = {}
    estimates_s = {
        name: _estimate_from_history(default_s, timing_history.get(name, []))
        for name, default_s in base_estimates_s.items()
    }
    steps: list[dict[str, object]] = []
    dep_checks: dict[str, dict[str, object]] = {}

    preflight_items: list[tuple[str, bool, str]] = []
    preflight_items.append(("python", *check_dependency([sys.executable, "--version"], root)))
    if not args.skip_configure or not args.skip_build:
        preflight_items.append(("cmake", *check_dependency(["cmake", "--version"], root)))
    if not args.skip_ctest:
        preflight_items.append(("ctest", *check_dependency(["ctest", "--version"], root)))
    preflight_items.append(("vector_db_dir", vector_db_dir.exists(), "ok" if vector_db_dir.exists() else "missing folder"))
    preflight_items.append(("generator_script", (root / "scripts" / "generate_synthetic_embeddings.py").exists(), "ok" if (root / "scripts" / "generate_synthetic_embeddings.py").exists() else "missing script"))
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

    def run_stage(name: str, cmd: list[str], cwd: Path, estimate_s: int, category: str) -> str:
        out, elapsed = run_stream_step(name, cmd, cwd, estimate_s)
        run_actual_s[category] = run_actual_s.get(category, 0.0) + elapsed
        steps.append(
            {
                "step": len(steps) + 1,
                "label": name,
                "command": cmd,
                "elapsed_s": round(elapsed, 6),
            }
        )
        return out

    try:
        if data_dir.exists() and not args.keep_data:
            shutil.rmtree(data_dir)

        if not args.skip_configure:
            run_stage(
                "Configure (CMake)",
                ["cmake", "-S", str(vector_db_dir), "-B", str(build_dir)],
                root,
                estimates_s["Configure (CMake)"],
                "Configure (CMake)",
            )
            timing_history.setdefault("Configure (CMake)", []).append(run_actual_s["Configure (CMake)"])

        if not args.skip_build:
            run_stage(
                "Build",
                ["cmake", "--build", str(build_dir), "--config", "Release"],
                root,
                estimates_s["Build"],
                "Build",
            )
            timing_history.setdefault("Build", []).append(run_actual_s["Build"])

        if not cli_path.exists():
            raise RuntimeError(f"missing CLI binary at {cli_path}; build first with cmake")

        if not args.skip_ctest:
            run_stage(
                "CTest",
                ["ctest", "--test-dir", str(build_dir), "--output-on-failure"],
                root,
                estimates_s["CTest"],
                "CTest",
            )
            timing_history.setdefault("CTest", []).append(run_actual_s["CTest"])

        if not args.skip_generate:
            gen_cmd = [
                sys.executable,
                "scripts/generate_synthetic_embeddings.py",
                "--count",
                str(args.count),
                "--dim",
                str(args.dim),
                "--seed",
                str(args.gen_seed),
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
            run_stage(
                "Generate synthetic dataset",
                gen_cmd,
                root,
                estimates_s["Generate synthetic dataset"],
                "Generate synthetic dataset",
            )
            timing_history.setdefault("Generate synthetic dataset", []).append(run_actual_s["Generate synthetic dataset"])

        if not payloads_path.exists():
            raise RuntimeError(f"missing payloads file: {payloads_path}")
        payload_count = count_jsonl_rows(payloads_path)
        if payload_count < 12:
            raise RuntimeError("payload file has too few rows; expected at least 12")

        pipeline_started = time.perf_counter()
        run_stage("init store", [str(cli_path), "init", "--path", str(data_dir)], vector_db_dir, 5, "Pipeline CLI flow")
        run_stage(
            "bulk insert payloads",
            [str(cli_path), "bulk-insert", "--path", str(data_dir), "--input", str(payloads_path)],
            vector_db_dir,
            45,
            "Pipeline CLI flow",
        )
        run_stage("checkpoint", [str(cli_path), "checkpoint", "--path", str(data_dir)], vector_db_dir, 8, "Pipeline CLI flow")
        run_stage(
            "build initial clusters",
            [str(cli_path), "build-initial-clusters", "--path", str(data_dir), "--seed", str(args.seed)],
            vector_db_dir,
            90,
            "Pipeline CLI flow",
        )
        cluster_stats_text = run_stage(
            "read cluster stats",
            [str(cli_path), "cluster-stats", "--path", str(data_dir)],
            vector_db_dir,
            5,
            "Pipeline CLI flow",
        )
        cluster_stats = json.loads(cluster_stats_text)
        source_version = args.source_version if args.source_version else str(cluster_stats["version"])
        run_stage(
            "build second-level clusters",
            [
                str(cli_path),
                "build-second-level-clusters",
                "--path",
                str(data_dir),
                "--seed",
                str(args.seed),
                "--source-version",
                source_version,
            ],
            vector_db_dir,
            120,
            "Pipeline CLI flow",
        )
        timing_history.setdefault("Pipeline CLI flow", []).append(time.perf_counter() - pipeline_started)

        second_level_doc = (
            data_dir
            / "clusters"
            / "initial"
            / f"v{source_version}"
            / "second_level_clustering"
            / "SECOND_LEVEL_CLUSTERING.json"
        )
        if not second_level_doc.exists():
            raise RuntimeError(f"missing second-level summary: {second_level_doc}")
        second_level = json.loads(second_level_doc.read_text(encoding="utf-8"))

        print("\n=== Final Cluster Summary ===")
        print(
            "first-layer: "
            f"chosen_k={cluster_stats.get('chosen_k')} "
            f"k_range=[{cluster_stats.get('k_min')},{cluster_stats.get('k_max')}] "
            f"vectors={cluster_stats.get('vectors_indexed')} "
            f"backend={cluster_stats.get('gpu_backend')} "
            f"tensor_core={cluster_stats.get('tensor_core_enabled')}"
        )
        print(
            "second-layer: "
            f"parent_centroids={second_level.get('total_parent_centroids')} "
            f"processed={second_level.get('processed_centroids')} "
            f"skipped={second_level.get('skipped_centroids')} "
            f"vectors_per_second={second_level.get('vectors_per_second')}"
        )
        for row in second_level.get("centroids", []):
            if row.get("processed"):
                print(
                    f"second-layer-centroid: id={row.get('centroid_id')} "
                    f"chosen_k={row.get('chosen_k')} "
                    f"vectors={row.get('vectors_indexed')} "
                    f"cuda={row.get('used_cuda')} tensor_core={row.get('tensor_core_enabled')}"
                )
            else:
                print(
                    f"second-layer-centroid: id={row.get('centroid_id')} "
                    f"chosen_k=0 skipped_reason={row.get('skipped_reason', 'not_processed')}"
                )

        total_elapsed_s = sum(float(step["elapsed_s"]) for step in steps)
        report = {
            "total_elapsed_s": round(total_elapsed_s, 6),
            "steps": steps,
            "dependency_preflight": dep_checks,
            "meta": {
                "payload_count": payload_count,
                "data_dir": str(data_dir),
                "payloads_path": str(payloads_path),
                "seed": str(args.seed),
                "source_version": int(source_version),
                "second_level_summary": str(second_level_doc),
            },
            "first_layer": cluster_stats,
            "second_layer": second_level,
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote report: {json_out}")

    except (RuntimeError, AssertionError, json.JSONDecodeError) as e:
        print(f"[FAIL] {e}")
        return 1
    finally:
        _print_timing_report(estimates_s, run_actual_s)
        try:
            _save_timings(root / TIMINGS_FILE, timing_history)
        except OSError:
            pass

    if not args.keep_data and data_dir.exists():
        shutil.rmtree(data_dir)
    print("\n[PASS] Pipeline test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

