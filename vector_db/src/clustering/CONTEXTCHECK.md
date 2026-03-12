# CONTEXTCHECK

- **Folder:** `vector_db/src/clustering`
- **Last run:** `2026-03-10`
- **Checker:** `context-folder-audit-batch`
- **Scope:** `non-recursive immediate files only`
- **CONTEXT.md dependency:** `disabled`

## Findings
- Reviewed `4` immediate files in this folder.
- No discrepancies detected under immediate-file audit rules.

## Status
- **Result:** `PASS`
- **Issue count:** `0`

## File Summaries
- `elbow_search.cpp`:
  - Implements spherical k-means fitting and binary-guided elbow selection over integer `k` candidates.
  - Builds stage-A/stage-B grids, supports optional pruning windows, and tracks early-stop reasons plus trace points.
  - Integrates INT8 search precision for elbow evaluation while preserving FP16 refit for the chosen final model.
  - Performs guard checks for CUDA availability, norm bounds, buffer shapes, and requested precision constraints.
  - Populates telemetry fields used downstream by cluster stats and manifest reporting.
  - related files: `id_estimator.cpp`, `stability.cpp`, `spherical_kmeans_cuda.cu`, `../../include/vector_db/clustering.hpp`, `../vector_store.cpp`

- `id_estimator.cpp`:
  - Estimates intrinsic dimensionality from sampled vectors using nearest/second-nearest neighbor distance ratios.
  - Computes quantile-based `m_low`/`m_high` statistics and derives clustering search bounds (`k_min`, `k_max`).
  - Uses randomized sampling with deterministic seeding for reproducibility and bounded runtime.
  - Applies guardrails for minimum sample size and finite metric handling when local estimates are unstable.
  - Returns normalized search bounds that feed elbow candidate generation.
  - related files: `elbow_search.cpp`, `../../include/vector_db/clustering.hpp`, `../vector_store.cpp`

- `spherical_kmeans_cuda.cu`:
  - Provides CUDA/cuBLASLt kernels and cache management for scoring, assignment, centroid reduction, and top-m extraction.
  - Implements FP16 and native INT8 scoring paths, including symmetric INT8 quantization and INT32 accumulation/dequantization.
  - Manages reusable GPU buffers through `GpuContextCache` and validates Ampere capability for INT8 tensor execution.
  - Exposes CUDA entry points consumed by clustering flow (`cuda_kmeans_iteration_top1`, `cuda_topm_from_centroids`, etc.).
  - Returns explicit errors on unsupported precision/hardware paths in line with GPU-only clustering policy.
  - related files: `elbow_search.cpp`, `../../include/vector_db/clustering.hpp`, `../vector_store.cpp`

- `stability.cpp`:
  - Evaluates clustering stability across repeated seeded runs using NMI, Jaccard overlap, and centroid drift metrics.
  - Supports adaptive stopping and optional parallel run execution while preserving deterministic run ordering.
  - Builds metrics from model comparisons and computes pass/fail based on configured quality thresholds.
  - Uses packed vector buffers when provided to avoid repeated host repacking overhead.
  - Produces stability outputs consumed by artifact writing and cluster health reporting.
  - related files: `elbow_search.cpp`, `id_estimator.cpp`, `../../include/vector_db/clustering.hpp`, `../vector_store.cpp`
