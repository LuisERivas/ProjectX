from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue


def _fmt_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def _progress_bar(elapsed_s: float, estimate_s: float, width: int = 24) -> str:
    if estimate_s <= 0:
        return "[unknown ETA]"
    ratio = min(1.0, elapsed_s / estimate_s)
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    eta_s = max(0.0, estimate_s - elapsed_s)
    pct = int(ratio * 100.0)
    return f"[{bar}] {pct:3d}% elapsed {_fmt_duration(elapsed_s)} eta {_fmt_duration(eta_s)}"


def run_step(name: str, cmd: list[str], cwd: Path, estimate_s: int) -> None:
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

        rc = proc.poll()
        now = time.monotonic()
        if rc is None and now - last_tick >= 1.0:
            elapsed = now - started
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

    estimates_s = {
        "Configure (CMake)": 45,
        "Build": 180,
        "CTest": 90,
        "Generate synthetic dataset": 20,
        "CLI smoke": 240,
        "Phase3 benchmark sanity": 60,
    }

    try:
        run_step(
            "Configure (CMake)",
            ["cmake", "-S", str(vector_db_dir), "-B", str(build_dir)],
            cwd=root,
            estimate_s=estimates_s["Configure (CMake)"],
        )
        run_step(
            "Build",
            ["cmake", "--build", str(build_dir), "--config", "Release"],
            cwd=root,
            estimate_s=estimates_s["Build"],
        )
        run_step(
            "CTest",
            ["ctest", "--test-dir", str(build_dir), "--output-on-failure"],
            cwd=root,
            estimate_s=estimates_s["CTest"],
        )
        run_step(
            "Generate synthetic dataset",
            [sys.executable, "scripts/generate_synthetic_embeddings.py"],
            cwd=root,
            estimate_s=estimates_s["Generate synthetic dataset"],
        )
        run_step(
            "CLI smoke",
            [sys.executable, "tests/smoke_cli.py"],
            cwd=vector_db_dir,
            estimate_s=estimates_s["CLI smoke"],
        )
        run_step(
            "Phase3 benchmark sanity",
            [sys.executable, "tests/benchmark_phase3.py"],
            cwd=vector_db_dir,
            estimate_s=estimates_s["Phase3 benchmark sanity"],
        )
    except FileNotFoundError as e:
        print(f"[FAIL] Missing tool: {e}")
        print("Install required tools (cmake, ctest, C++ compiler) and retry.")
        return 1
    except RuntimeError as e:
        print(f"[FAIL] {e}")
        return 1

    print("\n[PASS] Vector DB Phase 3 checks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

