from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_step(name: str, cmd: list[str], cwd: Path) -> None:
    print(f"\n== {name} ==")
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    vector_db_dir = root / "vector_db"
    build_dir = vector_db_dir / "build"

    print("Vector DB Phase 3 validation (Initial Clustering + CUDA path)")
    print("- Includes CRUD, tombstone, restart persistence checks")
    print("- Includes manifest.json + dirty_ranges.json reload checks")
    print("- Includes WAL append, replay, checkpoint, and malformed-tail tolerance checks")
    print("- Includes ID estimation, binary elbow k selection, stability checks, and clustering artifacts")

    try:
        run_step(
            "Configure (CMake)",
            ["cmake", "-S", str(vector_db_dir), "-B", str(build_dir)],
            cwd=root,
        )
        run_step(
            "Build",
            ["cmake", "--build", str(build_dir), "--config", "Release"],
            cwd=root,
        )
        run_step(
            "CTest",
            ["ctest", "--test-dir", str(build_dir), "--output-on-failure"],
            cwd=root,
        )
        run_step(
            "CLI smoke",
            [sys.executable, "tests/smoke_cli.py"],
            cwd=vector_db_dir,
        )
        run_step(
            "Phase3 benchmark sanity",
            [sys.executable, "tests/benchmark_phase3.py"],
            cwd=vector_db_dir,
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

