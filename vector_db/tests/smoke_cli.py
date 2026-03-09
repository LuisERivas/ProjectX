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


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main() -> int:
    build_dir = Path("build")
    bin_name = "vectordb_cli.exe" if sys.platform.startswith("win") else "vectordb_cli"
    cli = build_dir / bin_name
    if not cli.exists():
        raise RuntimeError(f"missing CLI binary at {cli}; build first with cmake")

    data_dir = Path("smoke_data")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    run([str(cli), "init", "--path", str(data_dir)])
    payloads_path = Path("synthetic_dataset_10k_fp16/insert_payloads.jsonl")
    if not payloads_path.exists():
        raise RuntimeError(
            f"missing synthetic payloads file: {payloads_path}. "
            "Run scripts/generate_synthetic_embeddings.py first."
        )
    payload_count = count_jsonl_rows(payloads_path)
    if payload_count < 12:
        raise RuntimeError("synthetic payloads file has too few rows; expected at least 12")
    run([str(cli), "bulk-insert", "--path", str(data_dir), "--input", str(payloads_path)])
    run([str(cli), "get", "--path", str(data_dir), "--id", "100"])
    run([str(cli), "update-meta", "--path", str(data_dir), "--id", "100", "--meta", '{"kind":"b","x":1}'])
    run([str(cli), "delete", "--path", str(data_dir), "--id", "100"])
    wal_before_checkpoint = json.loads(run([str(cli), "wal-stats", "--path", str(data_dir)]))
    assert wal_before_checkpoint["wal_entries"] >= 3
    stats_out = run([str(cli), "stats", "--path", str(data_dir)])
    parsed = json.loads(stats_out)
    assert parsed["total_rows"] == payload_count
    assert parsed["tombstone_rows"] == 1
    assert (data_dir / "manifest.json").exists()
    assert (data_dir / "dirty_ranges.json").exists()
    assert (data_dir / "wal.log").exists()

    run([str(cli), "checkpoint", "--path", str(data_dir)])
    wal_after_checkpoint = json.loads(run([str(cli), "wal-stats", "--path", str(data_dir)]))
    assert wal_after_checkpoint["wal_entries"] == 0
    assert wal_after_checkpoint["checkpoint_lsn"] >= wal_before_checkpoint["last_lsn"]

    run([str(cli), "build-initial-clusters", "--path", str(data_dir), "--seed", "9001"])
    cluster_stats = json.loads(run([str(cli), "cluster-stats", "--path", str(data_dir)]))
    cluster_health = json.loads(run([str(cli), "cluster-health", "--path", str(data_dir)]))
    assert cluster_stats["available"] is True
    assert cluster_stats["chosen_k"] >= cluster_stats["k_min"]
    assert cluster_stats["chosen_k"] <= cluster_stats["k_max"]
    assert cluster_health["available"] is True
    assert (data_dir / "clusters" / "initial" / "cluster_manifest.json").exists()

    # Restart behavior: invoke fresh process again and verify get still works.
    get_after_restart = run([str(cli), "get", "--path", str(data_dir), "--id", "100"])
    assert '"deleted": true' in get_after_restart
    stats_after_restart = json.loads(run([str(cli), "stats", "--path", str(data_dir)]))
    assert stats_after_restart["total_rows"] == payload_count
    assert stats_after_restart["tombstone_rows"] == 1
    assert stats_after_restart["dirty_ranges"] >= 3
    cluster_stats_after = json.loads(run([str(cli), "cluster-stats", "--path", str(data_dir)]))
    assert cluster_stats_after["version"] == cluster_stats["version"]

    shutil.rmtree(data_dir)
    print("smoke_cli: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

