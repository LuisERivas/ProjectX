# Test Gate Contract

## Purpose

Define mandatory automated gates for M1 readiness.

## Gate Set

- **G1 Correctness:** No critical correctness regressions.
- **G2 Durability:** WAL/checkpoint/replay pass crash-recovery scenarios.
- **G3 Contract Stability:** CLI/API/artifact schemas match contract docs.
- **G4 Performance:** Throughput/latency within agreed M1 budget for target hardware.
  - Section 14 wiring mode: `G4` is a mandatory hard gate in `full` profile, with Jetson Orin threshold enforcement using baseline-derived limits.
  - Hard mode statuses: `pass`, `fail_threshold`, `fail_env`.
- **G5 Compliance:** Required stages show `compliance_status=pass`.
- **G6 Fail-Fast:** Non-compliant required stages fail explicitly (no silent downgrade).
- **G7 Terminal Trace:** Required stage events are complete, ordered, and timing-valid.

## Required Validation Areas

- Exact-only query behavior.
- No metadata filter/ranking paths in M1.
- Top/Mid/Lower/Final stage ordering.
- Lower-layer gate semantics (`continue` vs `stop`).
- Final-layer eligibility semantics (only gate-stop leaf datasets).
- No cross-centroid mixing in restricted stages.
- Artifact existence and schema validity.
- Binary assignment artifact decoding/validation (layout, endianness, row-count, ordering, uniqueness).
- `id_estimate.bin` decoding/validation (record size, bounds invariants, non-zero values).
- End-of-pipeline `k_search_bounds_batch.bin` validation (row count, sort order, per-level semantics, bounds invariants).
- End-of-pipeline `post_cluster_membership.bin` validation (record count equals live embeddings, sorted/unique IDs, per-level ID mapping resolvability).
- Single-process composite full pipeline parity versus legacy stage-by-stage orchestration (same stage order, deterministic fail propagation, and equivalent final manifests/cluster stats).
- Per-cluster final artifact reconciliation with aggregate summary.
- CUDA critical-path best-practice compliance validation (parallelized critical path, minimized host-device transfers, launch configuration/utilization checks, coalesced global-memory access checks, redundant global-access reduction checks, divergence-aware kernel checks).
- Precision artifact ID-alignment validation (duplicate ID rejection, sorted-order validation, cardinality match across FP32/FP16/INT8 variants, exact ID-membership match across variants).
- Terminal alignment reporting validation (required fields present: `source_embedding_artifact`, `compute_precision`, `alignment_check_status`, `alignment_mismatch_count`, and `alignment_failure_reason` on fail).
- Fail-fast/regeneration enforcement for alignment failures (stage must regenerate derived artifacts or fail explicitly; stage success on unresolved alignment failure is disallowed).

## Terminal Trace Assertions

- Every required stage has `stage_start`.
- Every started stage has exactly one terminal event.
- Timing fields exist and are valid.
- `pipeline_elapsed_ms` is monotonic.
- Required per-centroid/per-cluster sub-events are present.

## Contract-Test Mapping Requirement

Each test suite must map to one or more contract sections in:

- `M1_SCOPE_CONTRACT.md`
- `CLI_CONTRACT.md`
- `ARTIFACT_CONTRACT.md`
- `TERMINAL_EVENT_CONTRACT.md`
- `COMPLIANCE_CONTRACT.md`

