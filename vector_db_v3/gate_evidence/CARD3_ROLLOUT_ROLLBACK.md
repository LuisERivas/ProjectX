# Card 3 Rollout and Rollback Notes

## Rollout Conditions

- Build succeeds with Card 3 sources enabled.
- Targeted tests pass:
  - `vectordb_v3_kmeans_backend_parity_tests`
  - `vectordb_v3_kmeans_tie_break_determinism_tests`
  - `vectordb_v3_kmeans_backend_selection_tests`
  - `vectordb_v3_gpu_residency_tests`
  - `vectordb_v3_compliance_pass_tests`
  - `vectordb_v3_compliance_fail_fast_tests`
  - `vectordb_v3_terminal_event_contract_tests`
- Stage telemetry includes additive residency fields:
  - `residency_mode`
  - `gpu_residency_cache_hits`
  - `gpu_residency_cache_misses`
  - `gpu_residency_bytes_reused`
  - `gpu_residency_bytes_h2d_saved_est`
  - `gpu_residency_alloc_calls`

## Rollback Triggers

- Any deterministic parity regression in CPU/CUDA/Tensor tests.
- Any compliance truthfulness regression in `kernel_backend_path` or `tensor_core_active`.
- Any stage fail-fast semantic drift under required compliance policy.
- Material performance regression in top/mid stage latency under residency mode.

## Rollback Actions

1. Set `VECTOR_DB_V3_GPU_RESIDENCY_MODE=off` to disable Card 3 at runtime.
2. Keep existing CUDA/Tensor backend selection unchanged and verify fallback to prior behavior.
3. Re-run targeted tests and G1/G3/G5/G6/G7 gates after disabling residency.

## Local Validation Blockers Observed In This Session

- `cmake` is not available in PATH on this machine.
- `ctest` is not available in PATH on this machine.
- Because no fresh CLI binary could be built, A/B perf script execution was blocked (`missing cli binary`).
