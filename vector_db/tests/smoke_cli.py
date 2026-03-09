from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def make_vec() -> str:
    return ",".join(f"{(i * 0.01):.6f}" for i in range(1024))


def main() -> int:
    build_dir = Path("build")
    bin_name = "vectordb_cli.exe" if sys.platform.startswith("win") else "vectordb_cli"
    cli = build_dir / bin_name
    if not cli.exists():
        raise RuntimeError(f"missing CLI binary at {cli}; build first with cmake")

    data_dir = Path("smoke_data")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    vec = make_vec()
    run([str(cli), "init", "--path", str(data_dir)])
    run([str(cli), "insert", "--path", str(data_dir), "--id", "100", "--vec", vec, "--meta", '{"kind":"a"}'])
    run([str(cli), "get", "--path", str(data_dir), "--id", "100"])
    run([str(cli), "update-meta", "--path", str(data_dir), "--id", "100", "--meta", '{"kind":"b","x":1}'])
    run([str(cli), "delete", "--path", str(data_dir), "--id", "100"])
    stats_out = run([str(cli), "stats", "--path", str(data_dir)])
    parsed = json.loads(stats_out)
    assert parsed["total_rows"] == 1
    assert parsed["tombstone_rows"] == 1
    assert (data_dir / "manifest.json").exists()
    assert (data_dir / "dirty_ranges.json").exists()

    # Restart behavior: invoke fresh process again and verify get still works.
    get_after_restart = run([str(cli), "get", "--path", str(data_dir), "--id", "100"])
    assert '"deleted": true' in get_after_restart
    stats_after_restart = json.loads(run([str(cli), "stats", "--path", str(data_dir)]))
    assert stats_after_restart["total_rows"] == 1
    assert stats_after_restart["tombstone_rows"] == 1
    assert stats_after_restart["dirty_ranges"] >= 3

    shutil.rmtree(data_dir)
    print("smoke_cli: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

