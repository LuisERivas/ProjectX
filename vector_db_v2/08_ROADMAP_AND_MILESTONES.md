# 08 Roadmap and Milestones

## Purpose

Provide implementation sequencing, delivery scope, and risk controls for Vector DB v2.

## Inputs/Dependencies

- [01 Product and Requirements](./01_PRODUCT_AND_REQUIREMENTS.md)
- [02 System Architecture](./02_SYSTEM_ARCHITECTURE.md)
- [07 Testing and Validation](./07_TESTING_AND_VALIDATION.md)

## Milestone Plan

- **M0: Foundation**
  - Finalize docs and contract baselines.
  - Define test harness skeleton and artifact schema validation helpers.

- **M1: Durable Core + Exact Search + Single Active Hierarchical Clustering State**
  - Mandatory M1 completion criteria:
  - Implement core store lifecycle, WAL/checkpoint, manifests.
  - Implement exact vector-similarity search path and baseline CLI commands.
  - Keep query record/response model minimal (`embedding_id`, `vector` in storage; ID + score ranking outputs).
  - Exclude ANN, metadata-based filtering/ranking, and artifact/version history from M1.
  - Implement Top -> Mid -> Lower -> Final clustering using one active artifact/state set.
  - Ensure Mid runs once globally, Lower uses per-centroid continued-processing gate, and Final runs DBSCAN only for gate-fail (`stop`) Lower-layer leaf centroid datasets that are non-empty and DBSCAN-valid (no cross-centroid mixing).
  - Enforce minimal Final-layer per-centroid artifact contract in M1 (`manifest.json`, `labels.json`, `cluster_summary.json`) plus aggregate `FINAL_LAYER_DBSCAN.json`.
  - Enforce CUDA-required execution on Ampere-class GPUs for performance-critical stages.
  - Enforce Tensor Core utilization for eligible INT8/FP16 kernels as a compliance target.
  - Enforce C++-first (C++/CUDA) implementation for hot-path compute kernels and orchestration.
  - Enforce fail-fast non-compliance handling for required hardware/runtime path violations.
  - Enforce terminal stage-trace + timing contract (start/end/fail events, per-stage elapsed, cumulative pipeline elapsed) across Top/Mid/Lower/Final, required per-centroid Lower-layer jobs, and required per-centroid Final-layer DBSCAN jobs.
  - Pass replay and contract tests.



## Risks and Mitigations

- **Risk:** Schema drift breaks scripts.
  - Mitigation: contract tests on JSON keys and strict additive-change policy.
- **Risk:** GPU path variability causes non-deterministic behavior.
  - Mitigation: deterministic seed policy + explicit runtime/precision telemetry.
- **Risk:** Durability regressions under crash scenarios.
  - Mitigation: replay/checkpoint integration tests as mandatory gates.

## Decisions and Rationale

- **Decision D-ROADMAP-001:** Sequence correctness/durability before ANN features.
  - Why: stable core is prerequisite for safe optimization.
  - Rejected alternative: parallel ANN and durability workstreams from day one. Rejected due to increased integration risk.

- **Decision D-ROADMAP-002:** Tie milestone completion to objective gates.
  - Why: prevents “feature complete but unvalidated” handoffs.
  - Rejected alternative: date-only milestone exits. Rejected due to poor quality predictability.

## Open Questions


## Exit Criteria

- Each milestone has explicit done criteria and measurable gates.
- Team can start M1 implementation without unresolved architecture blockers.
