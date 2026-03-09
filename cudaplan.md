# Revised 6-Phase Plan (Hierarchical Multi-Assignment + Per-Vector HNSW, Ampere-Aware)

## Phase 1 - FP32 Storage + CRUD + Metadata (Disk-First)
- Implement FP32 canonical vector store (`N x 1024`).
- CRUD ops: insert/delete/update metadata.
- Segment + manifest format with vector IDs and tombstones.
- Add "dirty range" tracking for future FP16 cache rebuilds.

### Ampere gate
- Define GPU staging format early (aligned row stride for coalesced loads).
- Ensure data layout supports vectorized FP16 conversion later.

## Phase 2 - WAL + Recovery + Startup Reload
- Append-only WAL for insert/delete/meta-update.
- Crash-safe replay and checkpointing.
- Startup: load FP32 segments, metadata, and index descriptors.

### Ampere gate
- WAL replay should mark exact dirty spans for async FP32->FP16 rebuild.
- Startup pipeline should support overlapped disk read + host preprocessing.

## Phase 3 - Hierarchical Multi-Assignment Clustering (Disk Index)
- Build hierarchical centroid tree (L0/L1/...).
- Assign each vector to top-`m` centroids per level (multi-assignment).
- Persist:
  - centroid vectors by level,
  - vector->centroid memberships,
  - centroid->posting lists.

### Ampere gate
- Serialize postings in GPU-friendly contiguous blocks.
- Keep centroid arrays contiguous and aligned for batched GEMM-like scoring.

## Phase 4 - GPU Hierarchical Cluster Search
- Query flow:
  1. FP32 query -> FP16 query buffer.
  2. Score against centroids level-by-level on GPU.
  3. Select top branches, merge multi-assignment postings.
  4. Build candidate set for rerank.
- Add candidate dedup and bounded candidate budget.

### Ampere gate
- Use FP16 input + FP32 accumulation.
- Tune stream concurrency (`H2D`, compute, `D2H`) and batch sizes for Orin Nano.
- Validate occupancy and memory throughput (not just kernel time).

## Phase 5 - Per-Vector HNSW (GPU-Assisted Build/Search)
- One graph node per vector (global graph or shard-global).
- Build with GPU-assisted neighbor scoring/pruning.
- Search:
  - use hierarchical cluster phase as optional entry-point seeding,
  - then HNSW traversal and top-k merge.

### Ampere gate
- Keep adjacency and vector blocks cache-friendly for repeated neighbor expansions.
- Separate kernels for frontier expansion and neighbor scoring.

## Phase 6 - GPU Dot-Product Core (Shared Primitive)
- Centralize similarity primitive used by phases 4 and 5.
- FP16 compute path with FP32 accumulate.
- Batched top-k kernels + stable tie handling.

### Ampere gate
- Explicit tensor-core path validation (WMMA/cuBLASLt or equivalent).
- Compare tensor-core path vs non-tensor path for correctness + throughput.
- Benchmark under sustained load/power mode to catch thermal throttling.

## Cross-Phase Validation Matrix (Required)

- Correctness:
  - CRUD/WAL/recovery deterministic
  - multi-assignment integrity (vector appears in expected memberships)
  - HNSW invariants (connectivity, degree bounds)
- Numerical:
  - FP16 inference vs FP32 reference tolerance thresholds
- Performance:
  - p50/p95 latency and QPS for batch sizes `{1, 8, 32, 64}`
  - throughput under sustained runs (thermal-aware)
- Resource:
  - VRAM footprint for FP16 caches + graph + postings
  - graceful behavior near memory limits

## Important note for Orin Nano
Yes, this revised plan explicitly accounts for Ampere CUDA/tensor utilization if the Ampere gates are enforced each phase (tensor-core path checks, coalesced layout, stream overlap, sustained thermal benchmarking). Without those gates, it will run on CUDA, but significant performance can be left on the table.

