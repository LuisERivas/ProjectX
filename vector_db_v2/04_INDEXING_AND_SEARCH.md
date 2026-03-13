# 04 Indexing and Search

## Purpose

Define M1 exact-only search behavior and query semantics for v2.

## Inputs/Dependencies

- [01 Product and Requirements](./01_PRODUCT_AND_REQUIREMENTS.md)
- [02 System Architecture](./02_SYSTEM_ARCHITECTURE.md)
- [06 API and CLI Contracts](./06_API_AND_CLI_CONTRACTS.md)

## Search Mode (M1)

- **Exact-only mode (required):**
  - Full score computation against live vectors.
  - Top-k ranking by vector similarity only.
  - Ground-truth reference behavior for correctness.
  - Performance-critical exact scoring implementation is C++/CUDA and must use CUDA-required execution on Ampere-class GPUs.

## Query Pipeline (Target)

1. Validate query vector dimension.
2. Score live vectors exactly with CUDA-required execution for performance-critical path.
3. Apply top-k ranking by vector similarity only.
4. Return minimal results required by contract (IDs and similarity scores).

## Out of Scope in M1

- ANN or other approximate/index-based retrieval stages.
- Metadata-based filtering.
- Metadata-influenced ranking.

## Scoring and Precision Strategy

- M1 numeric policy: embeddings are received/stored as FP32; optimal cluster-count estimation (k-selection) is computed in INT8; final clustering and assignment after k is selected are computed in FP16.
- INT8 is limited to k-selection stages (ID estimate / elbow / k-search logic).
- FP16 is used for final clustering and assignment once `k` is fixed.
- Query ranking remains exact-only and deterministic for same input + seed + hardware path.
- Tensor Cores are a primary target for eligible INT8/FP16 kernels, with compliance telemetry required.
- If required CUDA/Ampere/Tensor Core execution conditions are not met for performance-critical stages, execution must fail fast with explicit reason.

## Decisions and Rationale

- **Decision D-SEARCH-001:** Exact mode is mandatory baseline before ANN.
  - Why: provides correctness anchor and simpler incident debugging.
  - Rejected alternative: ANN-only from day one. Rejected due to harder correctness validation.

- **Decision D-SEARCH-002:** Keep CUDA-required execution for performance-critical high-dimensional scoring.
  - Why: aligns with workload profile and existing operational expectations.
  - Rejected alternative: CPU-default with optional GPU flag. Rejected due to inconsistent performance, operator surprise, and lack of fail-fast compliance guarantees.

## Open Questions


## Exit Criteria

- Query semantics and top-k behavior are unambiguous.
- Exact mode test vectors and expected results are defined in [07 Testing and Validation](./07_TESTING_AND_VALIDATION.md).
