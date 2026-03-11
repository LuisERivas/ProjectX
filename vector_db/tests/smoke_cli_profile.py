from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_timed(cmd: list[str]) -> tuple[str, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"elapsed_s: {elapsed_s:.3f}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout.strip(), elapsed_s


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile vectordb_cli smoke flow timing.")
    parser.add_argument("--data-dir", default="smoke_data_profile", help="Data directory for this profile run.")
    parser.add_argument(
        "--payloads",
        default="synthetic_dataset_10k_fp16/insert_payloads.jsonl",
        help="Insert payload JSONL path.",
    )
    parser.add_argument("--seed", default="9001", help="Seed for build-initial-clusters.")
    parser.add_argument(
        "--json-out",
        default="smoke_cli_profile_report.json",
        help="Write timing report JSON to this path.",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Do not delete profile data directory at the end.",
    )
    args = parser.parse_args()

    build_dir = Path("build")
    bin_name = "vectordb_cli.exe" if sys.platform.startswith("win") else "vectordb_cli"
    cli = build_dir / bin_name
    if not cli.exists():
        raise RuntimeError(f"missing CLI binary at {cli}; build first with cmake")

    data_dir = Path(args.data_dir)
    payloads_path = Path(args.payloads)
    json_out = Path(args.json_out)

    if data_dir.exists():
        shutil.rmtree(data_dir)

    if not payloads_path.exists():
        raise RuntimeError(
            f"missing synthetic payloads file: {payloads_path}. "
            "Run scripts/generate_synthetic_embeddings.py first."
        )

    payload_count = count_jsonl_rows(payloads_path)
    if payload_count < 12:
        raise RuntimeError("synthetic payloads file has too few rows; expected at least 12")

    steps: list[dict[str, object]] = []
    step_no = 0

    def do(label: str, cmd: list[str]) -> str:
        nonlocal step_no
        step_no += 1
        print(f"[profile step {step_no}] {label}")
        stdout, elapsed_s = run_timed(cmd)
        print(f"  elapsed: {elapsed_s:.3f}s")
        steps.append(
            {
                "step": step_no,
                "label": label,
                "command": cmd,
                "elapsed_s": round(elapsed_s, 6),
            }
        )
        return stdout

    do("init store", [str(cli), "init", "--path", str(data_dir)])
    do("bulk insert payloads", [str(cli), "bulk-insert", "--path", str(data_dir), "--input", str(payloads_path)])
    do("read row 100", [str(cli), "get", "--path", str(data_dir), "--id", "100"])
    do("update row 100 meta", [str(cli), "update-meta", "--path", str(data_dir), "--id", "100", "--meta", '{"kind":"b","x":1}'])
    do("delete row 100", [str(cli), "delete", "--path", str(data_dir), "--id", "100"])

    wal_before_checkpoint = json.loads(do("check WAL before checkpoint", [str(cli), "wal-stats", "--path", str(data_dir)]))
    assert wal_before_checkpoint["wal_entries"] >= 3

    stats_out = do("collect stats", [str(cli), "stats", "--path", str(data_dir)])
    parsed = json.loads(stats_out)
    assert parsed["total_rows"] == payload_count
    assert parsed["tombstone_rows"] == 1
    assert (data_dir / "manifest.json").exists()
    assert (data_dir / "dirty_ranges.json").exists()
    assert (data_dir / "wal.log").exists()

    do("checkpoint", [str(cli), "checkpoint", "--path", str(data_dir)])
    wal_after_checkpoint = json.loads(do("check WAL after checkpoint", [str(cli), "wal-stats", "--path", str(data_dir)]))
    assert wal_after_checkpoint["wal_entries"] == 0
    assert wal_after_checkpoint["checkpoint_lsn"] >= wal_before_checkpoint["last_lsn"]

    do("build initial clusters", [str(cli), "build-initial-clusters", "--path", str(data_dir), "--seed", str(args.seed)])
    cluster_stats = json.loads(do("read cluster stats", [str(cli), "cluster-stats", "--path", str(data_dir)]))
    cluster_health = json.loads(do("read cluster health", [str(cli), "cluster-health", "--path", str(data_dir)]))
    assert cluster_stats["available"] is True
    assert cluster_stats["chosen_k"] >= cluster_stats["k_min"]
    assert cluster_stats["chosen_k"] <= cluster_stats["k_max"]
    assert "gpu_backend" in cluster_stats
    assert "tensor_core_enabled" in cluster_stats
    assert "scoring_ms_total" in cluster_stats
    assert "scoring_calls" in cluster_stats
    assert cluster_health["available"] is True
    assert (data_dir / "clusters" / "initial" / "cluster_manifest.json").exists()

    get_after_restart = do("restart check get row 100", [str(cli), "get", "--path", str(data_dir), "--id", "100"])
    assert '"deleted": true' in get_after_restart
    stats_after_restart = json.loads(do("restart check stats", [str(cli), "stats", "--path", str(data_dir)]))
    assert stats_after_restart["total_rows"] == payload_count
    assert stats_after_restart["tombstone_rows"] == 1
    assert stats_after_restart["dirty_ranges"] >= 3
    cluster_stats_after = json.loads(do("restart check cluster stats", [str(cli), "cluster-stats", "--path", str(data_dir)]))
    assert cluster_stats_after["version"] == cluster_stats["version"]

    total_s = sum(float(s["elapsed_s"]) for s in steps)
    ranked = sorted(steps, key=lambda x: float(x["elapsed_s"]), reverse=True)

    print("\n=== CLI Smoke Profile Report ===")
    print(f"Total measured command time: {total_s:.3f}s")
    for s in ranked:
        pct = (float(s["elapsed_s"]) / total_s * 100.0) if total_s > 0 else 0.0
        print(f"- #{s['step']:>2} {s['label']}: {float(s['elapsed_s']):.3f}s ({pct:.1f}%)")

    report = {
        "total_elapsed_s": round(total_s, 6),
        "steps_ranked": ranked,
        "steps_in_order": steps,
        "meta": {
            "payload_count": payload_count,
            "data_dir": str(data_dir),
            "payloads_path": str(payloads_path),
            "seed": str(args.seed),
        },
    }
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote JSON report: {json_out}")

    if not args.keep_data and data_dir.exists():
        shutil.rmtree(data_dir)

    print("smoke_cli_profile: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
