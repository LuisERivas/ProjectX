from __future__ import annotations

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
TIMINGS_FILE = ".vector_db_phase3_timings.json"
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
        # Use p75 to avoid chronically underestimating variable steps.
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
        if est > 0:
            pct = (delta / est) * 100.0
            pct_text = f"{pct:+.1f}%"
        else:
            pct_text = "n/a"
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


def run_step(name: str, cmd: list[str], cwd: Path, estimate_s: int) -> float:
    print(f"\n== {name} ==")
    print(f"$ {' '.join(cmd)}")
    print(f"Estimated duration: ~{_fmt_duration(estimate_s)} (environment-dependent)")
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
                    eta_s = max(0.0, per_step * float(substep_total - completed))
                    total_estimate = elapsed + eta_s
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


def clear_old_test_paths(vector_db_dir: Path) -> None:
    stale_dirs = [
        vector_db_dir / "smoke_data",
        vector_db_dir / "tc_check_data",
    ]
    print("\n== Cleanup old test paths ==")
    for d in stale_dirs:
        if d.exists():
            print(f"- removing {d}")
            if d.is_dir():
                shutil.rmtree(d)
            else:
                d.unlink(missing_ok=True)
        else:
            print(f"- already clean {d}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    vector_db_dir = root / "vector_db"
    build_dir = vector_db_dir / "build"

    print("Vector DB Phase 3 validation (Initial Clustering + CUDA path)")
    print("- Includes CRUD, tombstone, restart persistence checks")
    print("- Includes manifest.json + dirty_ranges.json reload checks")
    print("- Includes WAL append, replay, checkpoint, and malformed-tail tolerance checks")
    print("- Includes ID estimation, binary elbow k selection, stability checks, and clustering artifacts")
    print("- Generates synthetic 10k FP16 embeddings and reuses them in smoke flow")

    base_estimates_s = {
        "Configure (CMake)": 45,
        "Build": 180,
        "CTest": 90,
        "Generate synthetic dataset": 20,
        "CLI smoke": 240,
    }
    timings_path = root / TIMINGS_FILE
    timing_history = _load_timings(timings_path)
    run_actual_s: dict[str, float] = {}
    estimates_s = {
        name: _estimate_from_history(default_s, timing_history.get(name, []))
        for name, default_s in base_estimates_s.items()
    }

    try:
        clear_old_test_paths(vector_db_dir)
        elapsed = run_step(
            "Configure (CMake)",
            ["cmake", "-S", str(vector_db_dir), "-B", str(build_dir)],
            cwd=root,
            estimate_s=estimates_s["Configure (CMake)"],
        )
        run_actual_s["Configure (CMake)"] = elapsed
        timing_history.setdefault("Configure (CMake)", []).append(elapsed)
        elapsed = run_step(
            "Build",
            ["cmake", "--build", str(build_dir), "--config", "Release"],
            cwd=root,
            estimate_s=estimates_s["Build"],
        )
        run_actual_s["Build"] = elapsed
        timing_history.setdefault("Build", []).append(elapsed)
        elapsed = run_step(
            "CTest",
            ["ctest", "--test-dir", str(build_dir), "--output-on-failure"],
            cwd=root,
            estimate_s=estimates_s["CTest"],
        )
        run_actual_s["CTest"] = elapsed
        timing_history.setdefault("CTest", []).append(elapsed)
        elapsed = run_step(
            "Generate synthetic dataset",
            [sys.executable, "scripts/generate_synthetic_embeddings.py"],
            cwd=root,
            estimate_s=estimates_s["Generate synthetic dataset"],
        )
        run_actual_s["Generate synthetic dataset"] = elapsed
        timing_history.setdefault("Generate synthetic dataset", []).append(elapsed)
        elapsed = run_step(
            "CLI smoke",
            [sys.executable, "tests/smoke_cli.py"],
            cwd=vector_db_dir,
            estimate_s=estimates_s["CLI smoke"],
        )
        run_actual_s["CLI smoke"] = elapsed
        timing_history.setdefault("CLI smoke", []).append(elapsed)
    except FileNotFoundError as e:
        print(f"[FAIL] Missing tool: {e}")
        print("Install required tools (cmake, ctest, C++ compiler) and retry.")
        return 1
    except RuntimeError as e:
        print(f"[FAIL] {e}")
        return 1
    finally:
        _print_timing_report(estimates_s, run_actual_s)
        try:
            _save_timings(timings_path, timing_history)
        except OSError:
            pass

    print("\n[PASS] Vector DB Phase 3 checks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

