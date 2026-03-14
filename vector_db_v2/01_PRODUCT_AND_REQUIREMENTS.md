# 01 Product and Requirements

## Purpose

Define what Vector DB v2 must do, what it explicitly will not do, and how success is measured.

## Inputs/Dependencies

- [README](./README.md)
- [02 System Architecture](./02_SYSTEM_ARCHITECTURE.md)
- [07 Testing and Validation](./07_TESTING_AND_VALIDATION.md)

## Product Goals

- Fast and reliable vector insert/search lifecycle for 1024-dim embeddings.
- Deterministic exact-only clustering pipeline with inspectable current-state artifacts.
- CUDA-required execution for heavy math with clear fail-fast behavior when hardware/runtime prerequisites are missing.
- Operational transparency through manifests, telemetry, and reproducible test scripts.

## Non-Goals (Initial v2)

- Multi-tenant auth/RBAC layer.
- Distributed sharding across multiple hosts.
- Online schema migration tooling for arbitrary historical formats.
- Cross-region replication.
- Metadata-based filtering and metadata-influenced ranking.
- ANN and other approximate/index-based retrieval paths.
- Clustering artifact/version history.

## M1 Scope Statement

- Record model is minimal: `{embedding_id, vector}`.
- Search and ranking are exact vector similarity only.
- Clustering is exact, hierarchical (Top layer -> Mid layer -> Lower layer -> Final layer), and deterministic.
- Metadata filtering is out of scope in M1 and metadata does not influence ranking.
- Cluster membership is pipeline output only, not a user-facing query filter in M1.
- Layer sequencing is strict: Top layer completes before Mid layer, Mid before Lower, and Final layer starts only after required Lower-layer gate evaluations and eligible per-centroid jobs are complete.
- Lower-layer continuation is governed by a per-centroid continued-processing split gate; outlier subgroup structure is used as evidence only, and approved continuation re-splits the full parent centroid dataset.
- Final layer executes passthrough finalization per eligible Lower-layer centroid dataset using that centroid's corresponding embeddings only; no cross-centroid mixing is allowed.
- Final-layer eligibility is gate-fail only: a centroid branch is eligible only when its Lower-layer continued-processing split gate decision is `stop` (`05` canonical rule).
- System maintains one active embedding corpus and one active clustering state.
- M1 numeric policy: embeddings are received/stored as FP32; optimal cluster-count estimation (k-selection) is computed in INT8; final clustering and assignment after k is selected are computed in FP16.
- M1 hardware/runtime policy: performance-critical stages are CUDA-required, Ampere-targeted, and Tensor Core-first where kernel eligibility permits.
- M1 deployment target wording: Ampere-class NVIDIA CUDA-capable deployment (including Ampere-class Jetson variants where applicable).
- M1 implementation policy: core hot paths are implemented in C++ (including CUDA/CUDA C++ kernels); higher-level scripting must not replace core compute kernels.

## Primary Use Cases

- Bulk ingest embeddings, then build Top-layer, Mid-layer, Lower-layer, and Final-layer clusters.
- Query nearest neighbors with deterministic exact vector-similarity ranking.
- Rebuild the current active clustering state from existing live embeddings.
- Run an end-to-end pipeline test that includes build, data generation, clustering, and artifact validation.

## Functional Requirements

- CRUD for vector records: insert, upsert policy, delete (tombstone).
- Batch ingestion interface for high-throughput imports.
- Cluster build commands:
  - build-top-clusters
  - build-mid-layer-clusters
  - build-lower-layer-clusters
  - build-final-layer-clusters
- Read-only diagnostics:
  - stats
  - wal-stats
  - cluster-stats
  - cluster-health
- Durable manifests and current clustering artifact outputs.
- M1 scope: maintain one active embedding corpus and one active clustering state; artifact/version history is out of scope until a later milestone.

## Non-Functional Requirements

- Determinism: same seed + same input yields same chosen_k and artifact structure.
- Durability: WAL + checkpoint model survives process crash without data loss after acknowledged write.
- Performance: CUDA path is required for performance-critical clustering/scoring from Top through Final stage orchestration; silent non-compliant path drops are disallowed.
- Hardware target: Ampere-class GPU behavior and Tensor Core utilization are mandatory optimization targets in M1 for eligible kernels.
- Fail-fast behavior: if required CUDA/Tensor Core prerequisites for compliant execution are missing, critical stage execution must fail with explicit machine-readable reason.
- Operability: all pipeline stages emit terminal-visible machine-readable stage lifecycle events by default (`stage_start`, `stage_end`/`stage_fail`) with per-stage elapsed timing and cumulative pipeline elapsed timing.
- Compatibility: CLI JSON remains backward-compatible for additive changes.

## SLO/SLA Targets (Initial)

- P0 correctness: no manifest/artifact corruption in tested flows.
- Pipeline pass rate: >= 99% in controlled CI/validation environments.
- Query correctness: exact vector-similarity ranking must be reference-correct.
- Clustering completion: must finish with explicit success/failure and diagnostics.

## Acceptance Criteria

- `scripts/pipeline_test.py` succeeds in default mode on target environment.
- Current active artifacts produced for Top/Mid/Lower/Final clustering outputs.
- `cluster-stats` and `cluster-health` parse as valid JSON and contain required contract fields.
- Reopen/replay tests validate WAL consistency and cluster manifest reloading.
- Terminal stage trace is complete and ordered for Top/Mid/Lower/Final stages with start/end/fail events, non-negative durations, and cumulative pipeline elapsed duration.
- Final-layer per-cluster outputs include canonical `assignments.json` contract compliance and required finalization status reporting.

## Decisions and Rationale

- **Decision D-REQ-001:** Keep CUDA-required clustering policy with hard-fail for unsupported hardware/runtime.
  - Why: avoids silent performance cliffs and ambiguous mixed-runtime behavior.
  - Rejected alternative: auto CPU fallback for clustering. Rejected due to hidden latency explosion and inconsistent run characteristics.

## Open Questions

- Must Lower-layer clustering always run, or remain opt-in per workload?
  - Lower-layer clustering continues only for centroid jobs that pass the continued-processing gate.

## Exit Criteria

- Functional and non-functional requirements map to concrete test gates in [07 Testing and Validation](./07_TESTING_AND_VALIDATION.md).
- Non-goals are stable and agreed before implementation starts.
