from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


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


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run second-level clustering smoke/validation and emit a report."
    )
    parser.add_argument("--data-dir", default="smoke_data_second_level")
    parser.add_argument("--payloads", default="synthetic_dataset_10k_fp16/insert_payloads.jsonl")
    parser.add_argument("--seed", default="9001")
    parser.add_argument("--json-out", default="vector_db/second_level_test_report.json")
    parser.add_argument("--keep-data", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    vector_db_dir = root / "vector_db"
    build_dir = vector_db_dir / "build"
    bin_name = "vectordb_cli.exe" if sys.platform.startswith("win") else "vectordb_cli"
    cli = build_dir / bin_name
    if not cli.exists():
        raise RuntimeError(f"missing CLI binary at {cli}; build first with cmake")

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = vector_db_dir / data_dir
    payloads_path = Path(args.payloads)
    if not payloads_path.is_absolute():
        payloads_path = vector_db_dir / payloads_path
    json_out = Path(args.json_out)
    if not json_out.is_absolute():
        json_out = root / json_out

    if data_dir.exists():
        shutil.rmtree(data_dir)
    if not payloads_path.exists():
        raise RuntimeError(f"missing payloads file: {payloads_path}")
    payload_count = count_jsonl_rows(payloads_path)
    if payload_count < 12:
        raise RuntimeError("payload file has too few rows; expected at least 12")

    steps: list[dict[str, object]] = []

    def do(label: str, cmd: list[str]) -> str:
        print(f"[step {len(steps) + 1}] {label}")
        out, elapsed_s = run_timed(cmd, vector_db_dir)
        steps.append(
            {
                "step": len(steps) + 1,
                "label": label,
                "command": cmd,
                "elapsed_s": round(elapsed_s, 6),
            }
        )
        return out

    do("init", [str(cli), "init", "--path", str(data_dir)])
    do("bulk-insert", [str(cli), "bulk-insert", "--path", str(data_dir), "--input", str(payloads_path)])
    do("build-initial-clusters", [str(cli), "build-initial-clusters", "--path", str(data_dir), "--seed", str(args.seed)])
    cluster_stats = json.loads(do("cluster-stats", [str(cli), "cluster-stats", "--path", str(data_dir)]))
    do(
        "build-second-level-clusters",
        [str(cli), "build-second-level-clusters", "--path", str(data_dir), "--seed", str(args.seed)],
    )

    second_level_doc = (
        data_dir
        / "clusters"
        / "initial"
        / f"v{cluster_stats['version']}"
        / "second_level_clustering"
        / "SECOND_LEVEL_CLUSTERING.json"
    )
    if not second_level_doc.exists():
        raise RuntimeError(f"missing second-level summary: {second_level_doc}")
    second_level = json.loads(second_level_doc.read_text(encoding="utf-8"))
    if second_level["processed_centroids"] + second_level["skipped_centroids"] != second_level["total_parent_centroids"]:
        raise RuntimeError("second-level centroid accounting mismatch")

    print("\nSecond-level centroid summary:")
    for row in second_level["centroids"]:
        print(
            f"- centroid={row['centroid_id']} processed={row['processed']} "
            f"vectors={row['vectors_indexed']} chosen_k={row['chosen_k']} "
            f"cuda={row['used_cuda']} tensor_core={row['tensor_core_enabled']}"
        )

    total_elapsed_s = sum(float(step["elapsed_s"]) for step in steps)
    report = {
        "total_elapsed_s": round(total_elapsed_s, 6),
        "steps": steps,
        "meta": {
            "payload_count": payload_count,
            "data_dir": str(data_dir),
            "seed": str(args.seed),
            "second_level_summary": str(second_level_doc),
        },
        "second_level": second_level,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote report: {json_out}")

    if not args.keep_data and data_dir.exists():
        shutil.rmtree(data_dir)
    print("test_vector_db_second_level: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

