from __future__ import annotations

import argparse
import json
import math
import random
import struct
from pathlib import Path


def _build_centroids(cluster_count: int, dim: int, rng: random.Random) -> list[list[float]]:
    centroids: list[list[float]] = []
    for _ in range(cluster_count):
        c = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        n = math.sqrt(sum(v * v for v in c)) or 1.0
        centroids.append([v / n for v in c])
    return centroids


def _make_vector(
    centroid: list[float],
    noise_std: float,
    normalized: bool,
    rng: random.Random,
) -> list[float]:
    vec = [c + rng.gauss(0.0, noise_std) for c in centroid]
    if normalized:
        n = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / n for v in vec]
    return vec


def generate_dataset(
    out_dir: Path,
    count: int,
    dim: int,
    seed: int,
    clusters: int,
    noise_std: float,
    normalized: bool,
    gpu: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = out_dir / "vectors.fp16bin"
    vectors_fp32_path = out_dir / "vectors.fp32bin"
    ids_path = out_dir / "ids.u64bin"
    meta_path = out_dir / "meta.jsonl"
    metadata_path = out_dir / "metadata.jsonl"
    cli_payloads_path = out_dir / "insert_payloads.jsonl"
    manifest_path = out_dir / "manifest.json"

    rng = random.Random(seed)
    centroids = _build_centroids(clusters, dim, rng)
    row_pack = struct.Struct("<" + ("e" * dim))
    row_unpack = struct.Struct("<" + ("e" * dim))
    row_pack_fp32 = struct.Struct("<" + ("f" * dim))
    id_pack = struct.Struct("<Q")

    cluster_counts = [0 for _ in range(clusters)]
    with (
        vectors_path.open("wb") as vf,
        vectors_fp32_path.open("wb") as v32f,
        ids_path.open("wb") as idf,
        meta_path.open("w", encoding="utf-8") as mjf,
        metadata_path.open("w", encoding="utf-8") as mf,
        cli_payloads_path.open("w", encoding="utf-8") as cf,
    ):
        if gpu:
            try:
                import cupy as cp  # type: ignore

                cp.random.seed(seed)
                cp_centroids = cp.asarray(centroids, dtype=cp.float32)
                ids_cp = cp.arange(count, dtype=cp.int64)
                cluster_ids = cp.random.randint(0, clusters, size=count, dtype=cp.int64)
                chosen = cp_centroids[cluster_ids]
                noise = cp.random.normal(0.0, noise_std, size=(count, dim)).astype(cp.float32)
                vectors = chosen + noise
                if normalized:
                    norms = cp.linalg.norm(vectors, axis=1, keepdims=True)
                    norms = cp.maximum(norms, 1e-12)
                    vectors = vectors / norms
                vectors_host = cp.asnumpy(vectors)
                cluster_ids_host = cp.asnumpy(cluster_ids)
                ids_host = cp.asnumpy(ids_cp)
                for i in range(count):
                    cid = int(cluster_ids_host[i])
                    cluster_counts[cid] += 1
                    vec = [float(x) for x in vectors_host[i]]
                    packed = row_pack.pack(*vec)
                    vf.write(packed)  # Quantizes to FP16 on write.
                    vec_fp16 = row_unpack.unpack(packed)
                    v32f.write(row_pack_fp32.pack(*vec))
                    rid = int(ids_host[i])
                    idf.write(id_pack.pack(rid))
                    vec_csv = ",".join(f"{v:.6f}" for v in vec_fp16)
                    meta_json = {"kind": "synthetic", "cluster": cid}
                    meta_text = json.dumps(meta_json, separators=(",", ":"))
                    mjf.write(meta_text + "\n")
                    mf.write(json.dumps({"id": rid, "cluster": cid}) + "\n")
                    cf.write(
                        json.dumps(
                            {
                                "id": rid,
                                "vec_csv": vec_csv,
                                "meta_json": meta_text,
                            }
                        )
                        + "\n"
                    )
                gpu = True
            except Exception:
                gpu = False

        for i in range(count):
            if gpu:
                break
            cid = rng.randrange(clusters)
            cluster_counts[cid] += 1
            vec = _make_vector(centroids[cid], noise_std, normalized, rng)
            packed = row_pack.pack(*vec)
            vf.write(packed)  # Quantizes to FP16 on write.
            vec_fp16 = row_unpack.unpack(packed)
            v32f.write(row_pack_fp32.pack(*vec))
            idf.write(id_pack.pack(i))
            vec_csv = ",".join(f"{v:.6f}" for v in vec_fp16)
            meta_json = {"kind": "synthetic", "cluster": cid}
            meta_text = json.dumps(meta_json, separators=(",", ":"))
            mjf.write(meta_text + "\n")
            mf.write(json.dumps({"id": i, "cluster": cid}) + "\n")
            # Format plugs directly into existing vectordb_cli insert usage:
            # vectordb_cli insert --id <id> --vec <vec_csv> --meta <meta_json>
            cf.write(
                json.dumps(
                    {
                        "id": i,
                        "vec_csv": vec_csv,
                        "meta_json": meta_text,
                    }
                )
                + "\n"
            )

    manifest = {
        "count": count,
        "dimension": dim,
        "dtype": "float16",
        "layout": "row_major",
        "normalized": normalized,
        "seed": seed,
        "clusters": clusters,
        "noise_std": noise_std,
        "files": {
            "vectors": vectors_path.name,
            "vectors_fp32": vectors_fp32_path.name,
            "ids": ids_path.name,
            "meta": meta_path.name,
            "metadata": metadata_path.name,
            "cli_payloads": cli_payloads_path.name,
        },
        "cluster_counts": cluster_counts,
        "gpu_generation_used": gpu,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic FP16 embeddings with fixed feature dimension."
    )
    parser.add_argument("--count", type=int, default=10_000, help="Number of embeddings.")
    parser.add_argument("--dim", type=int, default=1024, help="Feature dimension.")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed.")
    parser.add_argument("--clusters", type=int, default=32, help="Synthetic latent clusters.")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.04,
        help="Gaussian noise std added around centroids.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization before FP16 quantization.",
    )
    parser.add_argument(
        "--out-dir",
        default="vector_db/synthetic_dataset_10k_fp16",
        help="Output directory path.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Attempt GPU generation via CuPy (falls back to CPU if unavailable).",
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be > 0")
    if args.dim != 1024:
        raise SystemExit("--dim must be 1024 for this project")
    if args.clusters <= 0:
        raise SystemExit("--clusters must be > 0")
    if args.noise_std < 0.0:
        raise SystemExit("--noise-std must be >= 0")

    out_dir = Path(args.out_dir).resolve()
    generate_dataset(
        out_dir=out_dir,
        count=args.count,
        dim=args.dim,
        seed=args.seed,
        clusters=args.clusters,
        noise_std=args.noise_std,
        normalized=not args.no_normalize,
        gpu=args.gpu,
    )
    print(f"ok: wrote dataset to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

