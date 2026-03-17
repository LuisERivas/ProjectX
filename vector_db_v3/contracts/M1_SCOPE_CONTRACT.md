# M1 Scope Contract

## Purpose

Freeze M1 feature boundaries and non-negotiable runtime behavior for Vector DB v3.

## In Scope (M1)

- Exact vector search only.
- Hierarchical clustering pipeline with stages: Top -> Mid -> Lower -> Final.
- Single active clustering state on disk.
- Local filesystem durability (WAL + checkpoint + replay).
- C++/CUDA implementation for performance-critical paths.
- CUDA-required runtime policy for critical stages.
- CUDA kernel/runtime best-practice compliance for critical paths (parallelism, minimized host-device transfers, tuned launch config, coalesced memory access, reduced redundant global accesses, divergence-aware kernels).
- Binary-first artifact storage 
- Binary FP32 ingest feed is allowed for synthetic/perf workloads, with unchanged command and stage semantics.
- End-of-pipeline batch finalization step that writes consolidated k-search bounds across clustering stages.
- End-of-pipeline consolidated post-cluster membership artifact with one row per live embedding.

## Out of Scope (M1)

- ANN and any approximate retrieval mode.
- Metadata-based filtering.
- Metadata-influenced ranking.
- Multi-version artifact history as first-class retention policy.
- Multi-tenant auth/RBAC.
- Distributed sharding and cross-region replication.

## Canonical Record Model (M1)

- Storage record fields:
  - `embedding_id`
  - `vector`
- Query response fields:
  - `embedding_id`
  - `score`

## Artifact Encoding Policy (M1)

- Binary-first for data-plane artifacts (assignments, centroid payloads, bulk per-cluster outputs).
- JSON allowed only for small control-plane files:
  - manifests
  - aggregate summaries
  - compatibility/debug metadata

## Precision Policy (M1)

- Input embeddings are FP32.
- K-selection logic uses INT8-targeted kernels where eligible.
- Clustering/assignment stages after k selection use FP16-targeted kernels where eligible.
- Quantization metadata sidecar files are not required in M1.
- Precision artifact consistency in M1 is enforced by `embedding_id` alignment across FP32/FP16/INT8 artifacts.

## Final-Layer Eligibility (M1)

A Lower-layer centroid dataset is eligible for Final-layer processing only if:

1. The branch reached Lower-layer gate evaluation.
2. The final gate decision is `stop`.
3. The dataset is non-empty and valid for finalization input.

Gate decision `continue` is not eligible for Final layer.

## Compatibility Policy

- Contract changes must be additive unless explicitly labeled breaking.
- Any breaking change requires migration notes and contract version bump.

