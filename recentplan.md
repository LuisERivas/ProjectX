# CUDA Optimization Execution Plan (Ordered)

## Step 1 - Fuse scoring + top1 + centroid update on GPU (`#3`)
Goal:
- Execute each k-means iteration primarily on GPU so score, top1 assignment, and centroid update do not bounce through CPU.

Files:
- `vector_db/src/clustering/elbow_search.cpp`
- `vector_db/src/clustering/spherical_kmeans_cuda.cu`
- `vector_db/include/vector_db/clustering.hpp`

Tasks:
- Add CUDA fused iteration API (`cuda_kmeans_iteration_top1`) to compute score->top1->centroid update.
- Route CUDA-enabled k-means iterations in `run_kmeans_impl` through this fused path.
- Keep deterministic CPU fallback for non-CUDA or failure conditions.
- Preserve telemetry fields (`used_cuda`, `tensor_core_enabled`, `gpu_backend`, scoring timing).

Done criteria:
- CUDA path no longer requires CPU-side assignment/reduction inside each iteration.
- Existing clustering logic and tests remain valid.

Validation:
- `python scripts/test_vector_db_phase1.py`
- `python vector_db/tests/smoke_cli.py`



## Step 2 - GPU top-m/top-k final assignment (`#5`)
Goal:
- Remove CPU full-sort final assignment in CUDA mode.

Files:
- `vector_db/src/clustering/elbow_search.cpp`
- `vector_db/src/clustering/spherical_kmeans_cuda.cu`
- `vector_db/include/vector_db/clustering.hpp`

Tasks:
- Add CUDA top-m API (`cuda_topm_from_centroids`) and kernel path.
- Use GPU top-m path for final assignment when CUDA is available.
- Keep CPU fallback and preserve assignment semantics.

Done criteria:
- Final assignment avoids CPU sorting in CUDA mode.
- Output remains compatible with downstream stats/health/reporting.

Validation:
- `python scripts/test_vector_db_phase1.py`
- `python vector_db/tests/smoke_cli.py`



## Step 3 - Keep vectors/centroids/scores device-resident (`#2`)
Goal:
- Reduce repeated host-device copies across iterative clustering.

Files:
- `vector_db/src/clustering/spherical_kmeans_cuda.cu`
- `vector_db/src/clustering/elbow_search.cpp`

Tasks:
- Add reusable GPU cache/context for vectors, centroids, scores, labels, and workspace.
- Reuse buffers across repeated CUDA scoring and assignment calls.
- Keep host transfers to required outputs only.

Done criteria:
- Per-iteration allocation/copy pressure is reduced compared to prior path.
- Behavior remains unchanged functionally.

Validation:
- Compare pre/post `cluster-stats` (`scoring_ms_total`, `scoring_calls`, `used_cuda`).
- Optional manual perf check: `python vector_db/tests/benchmark_phase3.py`



## Step 4 - Optimize centroid reduction kernel (`#4`)
Goal:
- Improve centroid update performance and reduce write pressure in update stage.

Files:
- `vector_db/src/clustering/spherical_kmeans_cuda.cu`

Tasks:
- Introduce hierarchical/partial reduction kernels for centroid accumulation.
- Keep stable reduction fallback path.
- Normalize centroids on GPU after reduction.

Done criteria:
- Reduction kernel path remains correct and benchmark-stable.
- No regressions in clustering validity.

Validation:
- `python scripts/test_vector_db_phase1.py`
- Optional manual perf check: `python vector_db/tests/benchmark_phase3.py`



## Step 5 - Benchmark script uses bulk-insert (`#6`)
Goal:
- Eliminate repeated CLI process overhead in benchmark setup.

Files:
- `vector_db/tests/benchmark_phase3.py`

Tasks:
- Replace per-row `insert` loop with generated JSONL payload + `bulk-insert`.
- Preserve benchmark intent and reporting format.

Done criteria:
- Benchmark ingest/setup runs faster.
- Benchmark output schema remains unchanged.

Validation:
- Benchmark step removed from default runner; run manually only if needed:
- `python vector_db/tests/benchmark_phase3.py`



## Step 6 - Persistent GPU buffers / no malloc-free churn (`#1`)
Goal:
- Remove hot-loop `cudaMalloc/cudaFree` overhead for scoring and related buffers.

Files:
- `vector_db/src/clustering/spherical_kmeans_cuda.cu`

Tasks:
- Reuse cached device buffers and cuBLASLt handle/workspace when shapes fit.
- Keep safe reallocation when larger shapes are requested.

Done criteria:
- Repeated scoring calls avoid fresh allocation churn.
- Runtime and scoring telemetry improve or remain stable.

Validation:
- `python scripts/test_vector_db_phase1.py`



## Step 7 - Binary ingest path (`#7`)
Goal:
- Avoid JSON+CSV parse overhead for large ingest.

Files:
- `vector_db/cli/main.cpp`
- `scripts/generate_synthetic_embeddings.py`

Tasks:
- Add `bulk-insert-bin` command with `--vectors`, `--ids`, `--meta`.
- Validate row count and vector byte-size consistency.
- Preserve existing `bulk-insert` compatibility.

Done criteria:
- Binary ingest command works with generated data artifacts.
- Existing flows still function.

Validation:
- `vectordb_cli bulk-insert-bin --path <dir> --vectors <vectors.fp32bin> --ids <ids.u64bin> --meta <meta.jsonl>`
- `python vector_db/tests/smoke_cli.py`



## Step 8 - GPU-accelerated synthetic dataset generation (`#8`)
Goal:
- Add optional GPU generation mode for synthetic data.

Files:
- `scripts/generate_synthetic_embeddings.py`

Tasks:
- Add `--gpu` flag using CuPy when available.
- Keep output-compatible files (`insert_payloads.jsonl`, binary/vector artifacts).
- Gracefully fall back to CPU generation if GPU path is unavailable.

Done criteria:
- Script succeeds in both CPU-only and GPU-enabled environments.
- Output format remains compatible with existing tests and CLI.

Validation:
- `python scripts/generate_synthetic_embeddings.py`
- `python scripts/generate_synthetic_embeddings.py --gpu`

