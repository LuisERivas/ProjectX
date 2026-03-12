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


def run_step(step: int, total: int, label: str, cmd: list[str]) -> str:
    print(f"[step {step}/{total}] {label}", flush=True)
    return run(cmd)


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def require_keys(obj: dict[str, object], required: list[str], context: str) -> None:
    missing = [k for k in required if k not in obj]
    if missing:
        raise RuntimeError(
            f"{context} is missing expected keys: {missing}. "
            "This usually means the CLI binary is stale; rebuild vectordb_cli and retry."
        )


def main() -> int:
    total_steps = 16
    step = 0

    def do(label: str, cmd: list[str]) -> str:
        nonlocal step
        step += 1
        return run_step(step, total_steps, label, cmd)

    build_dir = Path("build")
    bin_name = "vectordb_cli.exe" if sys.platform.startswith("win") else "vectordb_cli"
    cli = build_dir / bin_name
    if not cli.exists():
        raise RuntimeError(f"missing CLI binary at {cli}; build first with cmake")

    data_dir = Path("smoke_data")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    do("init store", [str(cli), "init", "--path", str(data_dir)])
    payloads_path = Path("synthetic_dataset_10k_fp16/insert_payloads.jsonl")
    if not payloads_path.exists():
        raise RuntimeError(
            f"missing synthetic payloads file: {payloads_path}. "
            "Run scripts/generate_synthetic_embeddings.py first."
        )
    payload_count = count_jsonl_rows(payloads_path)
    if payload_count < 12:
        raise RuntimeError("synthetic payloads file has too few rows; expected at least 12")
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

    do("build initial clusters", [str(cli), "build-initial-clusters", "--path", str(data_dir), "--seed", "9001"])
    cluster_stats = json.loads(do("read cluster stats", [str(cli), "cluster-stats", "--path", str(data_dir)]))
    cluster_health = json.loads(do("read cluster health", [str(cli), "cluster-health", "--path", str(data_dir)]))
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
    assert (data_dir / "clusters" / "initial" / "cluster_manifest.json").exists()

    do(
        "build second-level clusters",
        [str(cli), "build-second-level-clusters", "--path", str(data_dir), "--seed", "9001"],
    )
    second_level_doc = data_dir / "clusters" / "initial" / f"v{cluster_stats['version']}" / "second_level_clustering" / "SECOND_LEVEL_CLUSTERING.json"
    assert second_level_doc.exists()
    second_level = json.loads(second_level_doc.read_text(encoding="utf-8"))
    assert second_level["processed_centroids"] + second_level["skipped_centroids"] == second_level["total_parent_centroids"]
    assert second_level["total_parent_centroids"] >= 1
    if second_level["processed_centroids"] > 0:
        first_processed = next(c for c in second_level["centroids"] if c["processed"] is True)
        assert "used_cuda" in first_processed
        assert "tensor_core_enabled" in first_processed
        assert "elbow_int8_search_enabled" in first_processed
        assert "elbow_scoring_precision" in first_processed

    # Restart behavior: invoke fresh process again and verify get still works.
    get_after_restart = do("restart check get row 100", [str(cli), "get", "--path", str(data_dir), "--id", "100"])
    assert '"deleted": true' in get_after_restart
    stats_after_restart = json.loads(do("restart check stats", [str(cli), "stats", "--path", str(data_dir)]))
    assert stats_after_restart["total_rows"] == payload_count
    assert stats_after_restart["tombstone_rows"] == 1
    assert stats_after_restart["dirty_ranges"] >= 3
    cluster_stats_after = json.loads(do("restart check cluster stats", [str(cli), "cluster-stats", "--path", str(data_dir)]))
    assert cluster_stats_after["version"] == cluster_stats["version"]
    cluster_stats_after2 = json.loads(do("restart check cluster stats repeated", [str(cli), "cluster-stats", "--path", str(data_dir)]))
    assert cluster_stats_after2["version"] == cluster_stats_after["version"]

    shutil.rmtree(data_dir)
    print("smoke_cli: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

