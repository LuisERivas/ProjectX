from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_stage(name: str, cmd: list[str], cwd: Path) -> str:
    t0 = time.time()
    print(f"[stage_start] {name}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr.rstrip())
        raise RuntimeError(f"stage failed: {name} ({elapsed:.3f}s)")
    print(f"[stage_end] {name}: {elapsed:.3f}s")
    return proc.stdout


def generate_payload(
    path: Path,
    rows: int,
    synthetic_clusters: int,
    noise_std: float,
    outlier_ratio: float,
) -> None:
    rnd = random.Random()
    dim = 1024
    clusters = max(1, min(synthetic_clusters, rows))
    outlier_count = int(rows * max(0.0, min(outlier_ratio, 0.95)))
    inlier_count = rows - outlier_count

    # Precompute random cluster centers for reproducible, diverse synthetic groups.
    centers: list[list[float]] = []
    for _ in range(clusters):
        center = [rnd.uniform(-1.0, 1.0) for _ in range(dim)]
        centers.append(center)

    with path.open("w", encoding="utf-8") as f:
        embedding_id = 1
        for i in range(inlier_count):
            cidx = i % clusters
            center = centers[cidx]
            vec = [center[j] + rnd.gauss(0.0, noise_std) for j in range(dim)]
            vals = [f"{v:.6f}" for v in vec]
            row = {"embedding_id": embedding_id, "vec_csv": ",".join(vals)}
            f.write(json.dumps(row) + "\n")
            embedding_id += 1

        # Add some outliers so clustering has harder edge-cases.
        for _ in range(outlier_count):
            vec = [rnd.uniform(-4.0, 4.0) for _ in range(dim)]
            vals = [f"{v:.6f}" for v in vec]
            row = {"embedding_id": embedding_id, "vec_csv": ",".join(vals)}
            f.write(json.dumps(row) + "\n")
            embedding_id += 1


def print_generation_summary(rows: int, synthetic_clusters: int, noise_std: float, outlier_ratio: float) -> None:
    outlier_count = int(rows * max(0.0, min(outlier_ratio, 0.95)))
    inlier_count = rows - outlier_count
    print("\n=== Embedding Generation Summary ===")
    print("Seed mode: system entropy (non-deterministic)")
    print(f"Total embeddings: {rows}")
    print(f"Synthetic cluster centers: {max(1, min(synthetic_clusters, rows))}")
    print(f"Inlier embeddings: {inlier_count}")
    print(f"Outlier embeddings: {outlier_count}")
    print(f"Inlier noise stddev: {noise_std}")
    print("====================================\n")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_unique_key_rows(path: Path, key: str) -> int:
    rows = read_json(path)
    if not isinstance(rows, list):
        return 0
    uniq = set()
    for row in rows:
        if isinstance(row, dict) and key in row:
            uniq.add(str(row[key]))
    return len(uniq)


def count_rows_by_key(path: Path, key: str) -> dict[str, int]:
    rows = read_json(path)
    out: dict[str, int] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict) or key not in row:
            continue
        k = str(row[key])
        out[k] = out.get(k, 0) + 1
    return out


def assert_no_empty_mid(mid_summary: object) -> None:
    if not isinstance(mid_summary, dict):
        raise RuntimeError("mid summary must be an object")
    per_top = mid_summary.get("per_top_centroid", [])
    if not isinstance(per_top, list):
        raise RuntimeError("mid summary per_top_centroid must be a list")
    for row in per_top:
        if not isinstance(row, dict):
            continue
        chosen_k = int(row.get("chosen_k", 0))
        produced = int(row.get("produced_mid_centroids", 0))
        if chosen_k <= 0 or produced != chosen_k:
            raise RuntimeError("mid no-empty invariant failed for at least one top parent")


def assert_no_empty_lower(lower_summary: object) -> None:
    if not isinstance(lower_summary, dict):
        raise RuntimeError("lower summary must be an object")
    gate_rows = lower_summary.get("gate_evaluations", [])
    if not isinstance(gate_rows, list):
        raise RuntimeError("lower summary gate_evaluations must be a list")

    continue_expected: dict[str, int] = {}
    continue_children: dict[str, int] = {}
    for row in gate_rows:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("centroid_id", ""))
        decision = str(row.get("decision", ""))
        reason = str(row.get("reason", ""))
        chosen_k = int(row.get("chosen_k", 0))
        dataset_size = int(row.get("dataset_size", 0))
        if reason == "depth_cap_reached":
            if dataset_size <= 0:
                raise RuntimeError("lower no-empty invariant failed: empty depth_cap_reached leaf")
            parent, _, _ = cid.rpartition("_")
            if parent:
                continue_children[parent] = continue_children.get(parent, 0) + 1
        if decision == "continue":
            continue_expected[cid] = chosen_k

    for parent, expected in continue_expected.items():
        if expected <= 0 or continue_children.get(parent, 0) != expected:
            raise RuntimeError(f"lower no-empty invariant failed for parent {parent}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--synthetic-clusters", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--outlier-ratio", type=float, default=0.03)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build_dir = root / "build"
    data_dir = root / "tmp_pipeline_data"
    payload = root / "tmp_pipeline_insert.jsonl"
    cli = build_dir / "vectordb_v2_cli.exe"
    if not cli.exists():
        cli = build_dir / "vectordb_v2_cli"

    if data_dir.exists():
        shutil.rmtree(data_dir)
    if payload.exists():
        payload.unlink()

    if not args.skip_build:
        run_stage("cmake configure", ["cmake", "-S", str(root), "-B", str(build_dir)], root)
        run_stage("cmake build", ["cmake", "--build", str(build_dir)], root)
        run_stage("ctest", ["ctest", "--test-dir", str(build_dir), "--output-on-failure"], root)

    if not cli.exists():
        raise RuntimeError("missing vectordb_v2_cli after build")

    generate_payload(
        payload,
        args.rows,
        args.synthetic_clusters,
        args.noise_std,
        args.outlier_ratio,
    )
    print_generation_summary(args.rows, args.synthetic_clusters, args.noise_std, args.outlier_ratio)

    run_stage("init", [str(cli), "init", "--path", str(data_dir)], root)
    bulk_text = run_stage(
        "bulk-insert",
        [str(cli), "bulk-insert", "--path", str(data_dir), "--input", str(payload), "--batch-size", str(args.batch_size)],
        root,
    )
    if not re.search(
        r"bulk-insert summary:\s+parse_ms=\d+(\.\d+)?\s+wal_ms=\d+(\.\d+)?\s+persist_ms=\d+(\.\d+)?\s+total_ms=\d+(\.\d+)?",
        bulk_text,
    ):
        raise RuntimeError("bulk-insert summary line missing or malformed")
    run_stage("checkpoint", [str(cli), "checkpoint", "--path", str(data_dir)], root)
    run_stage("build-top-clusters", [str(cli), "build-top-clusters", "--path", str(data_dir)], root)
    run_stage("build-mid-layer-clusters", [str(cli), "build-mid-layer-clusters", "--path", str(data_dir)], root)
    run_stage("build-lower-layer-clusters", [str(cli), "build-lower-layer-clusters", "--path", str(data_dir)], root)
    run_stage("build-final-layer-clusters", [str(cli), "build-final-layer-clusters", "--path", str(data_dir)], root)
    stats_text = run_stage("stats", [str(cli), "stats", "--path", str(data_dir)], root)
    cstats = run_stage("cluster-stats", [str(cli), "cluster-stats", "--path", str(data_dir)], root)
    chealth = run_stage("cluster-health", [str(cli), "cluster-health", "--path", str(data_dir)], root)

    top_assignments = data_dir / "clusters" / "current" / "assignments.json"
    mid_assignments = data_dir / "clusters" / "current" / "mid_layer_clustering" / "assignments.json"
    lower_summary_path = data_dir / "clusters" / "current" / "lower_layer_clustering" / "LOWER_LAYER_CLUSTERING.json"
    final_summary_path = data_dir / "clusters" / "current" / "final_layer_clustering" / "FINAL_LAYER_CLUSTERS.json"

    store_stats = json.loads(stats_text)
    lower_summary = read_json(lower_summary_path)
    final_summary = read_json(final_summary_path)
    mid_summary = read_json(data_dir / "clusters" / "current" / "mid_layer_clustering" / "MID_LAYER_CLUSTERING.json")
    cluster_stats_obj = json.loads(cstats)

    embedding_total = int(store_stats.get("live_rows", 0))
    top_cluster_count = count_unique_key_rows(top_assignments, "top_centroid_id")
    mid_cluster_count = count_unique_key_rows(mid_assignments, "mid_centroid_id")
    lower_cluster_count = len(lower_summary.get("leaf_datasets", [])) if isinstance(lower_summary, dict) else 0

    top_rows = count_rows_by_key(top_assignments, "top_centroid_id")
    if not top_rows:
        raise RuntimeError("top assignments missing")
    if int(cluster_stats_obj.get("chosen_k", 0)) != len(top_rows):
        raise RuntimeError("top no-empty invariant failed: unique top centroids != chosen_k")
    if any(count <= 0 for count in top_rows.values()):
        raise RuntimeError("top no-empty invariant failed: empty top centroid detected")

    mid_rows = count_rows_by_key(mid_assignments, "mid_centroid_id")
    if not mid_rows:
        raise RuntimeError("mid assignments missing")
    if any(count <= 0 for count in mid_rows.values()):
        raise RuntimeError("mid no-empty invariant failed: empty mid centroid detected")
    assert_no_empty_mid(mid_summary)
    assert_no_empty_lower(lower_summary)

    final_cluster_count = 0
    if isinstance(final_summary, dict):
        per_cluster = final_summary.get("per_cluster", [])
        if isinstance(per_cluster, list):
            final_cluster_count = len(
                {
                    str(row.get("final_cluster_id"))
                    for row in per_cluster
                    if isinstance(row, dict) and row.get("final_layer_output_status") == "written"
                }
            )
    total_clusters_all_levels = top_cluster_count + mid_cluster_count + lower_cluster_count + final_cluster_count

    print("\n=== Cluster Count Summary ===")
    print(f"Total embeddings: {embedding_total}")
    print(f"Top layer clusters: {top_cluster_count}")
    print(f"Mid layer clusters: {mid_cluster_count}")
    print(f"Lower layer clusters: {lower_cluster_count}")
    print(f"Final layer clusters: {final_cluster_count}")
    print(f"Total clusters (all levels): {total_clusters_all_levels}")
    print("=============================\n")

    out = {
        "data_dir": str(data_dir),
        "stats": store_stats,
        "cluster_stats": json.loads(cstats),
        "no_empty_cluster_checks": {
            "top": "pass",
            "mid": "pass",
            "lower": "pass",
        },
        "cluster_health": json.loads(chealth),
        "cluster_counts": {
            "total_embeddings": embedding_total,
            "top_layer_clusters": top_cluster_count,
            "mid_layer_clusters": mid_cluster_count,
            "lower_layer_clusters": lower_cluster_count,
            "final_layer_clusters": final_cluster_count,
            "total_clusters_all_levels": total_clusters_all_levels,
        },
        "artifacts": {
            "mid": str(data_dir / "clusters" / "current" / "mid_layer_clustering" / "MID_LAYER_CLUSTERING.json"),
            "lower": str(data_dir / "clusters" / "current" / "lower_layer_clustering" / "LOWER_LAYER_CLUSTERING.json"),
            "final": str(data_dir / "clusters" / "current" / "final_layer_clustering" / "FINAL_LAYER_CLUSTERS.json"),
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
