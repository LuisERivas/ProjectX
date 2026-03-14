# 07 Testing and Validation

## Purpose

Define test strategy and release gates for v2 to ensure correctness, durability, and operational reliability.

## Inputs/Dependencies

- [01 Product and Requirements](./01_PRODUCT_AND_REQUIREMENTS.md)
- [03 Storage and Durability](./03_STORAGE_AND_DURABILITY.md)
- [06 API and CLI Contracts](./06_API_AND_CLI_CONTRACTS.md)
- [08 Roadmap and Milestones](./08_ROADMAP_AND_MILESTONES.md)

## Test Pyramid

- **Unit tests**
  - Serialization/parsing, k-selection helpers, manifest/WAL parsing, edge-case math.
- **Integration tests**
  - End-to-end CLI flows over local filesystem data dir.
  - WAL replay, checkpoint truncation, reopen consistency.
- **System tests**
  - Full pipeline run including build, dataset generation, Top/Mid/Lower/Final stages, and artifact verification.
- **Performance tests**
  - Throughput, latency, and GPU telemetry regression checks.

## Core Validation Areas

- Data correctness (CRUD, tombstones).
- Durability and recovery (WAL, checkpoint, crash-like reopen paths).
- Clustering correctness (artifact existence, schema validity, deterministic fields).
- Hierarchical sequencing correctness (Top -> Mid -> Lower -> Final ordering constraints).
- Lower-layer continued-processing gate behavior (eligible vs skipped centroid jobs).
- Final-layer trigger correctness (runs only after Lower-layer completion).
- Final-layer isolation correctness (fail if Final-layer processing mixes embeddings across centroid datasets).
- Final-layer per-cluster execution coverage (eligible gate-fail Lower-layer centroid datasets each receive Final-layer processing).
- Final-layer eligibility correctness (gate `continue` branches are excluded; gate `stop` branches are included).
- Contract compatibility (`cluster-stats`, `cluster-health`, command exit semantics).
- GPU telemetry consistency and fail-fast behavior.
- Exact-only query behavior (no ANN path, no metadata filter/ranking path in M1).
- Hardware compliance for required stages (CUDA-required path, Tensor Core eligibility/activation, Ampere-class target, C++/CUDA hot-path implementation).

## Minimal Reproducible Pipeline Gate

- Build + test (`cmake`, `ctest`) must pass.
- Run `scripts/pipeline_test.py` equivalent for v2.
- Validate generated report JSON and required artifacts:
  - Top-layer artifacts under `clusters/current/`
  - Mid-layer summary under `mid_layer_clustering/MID_LAYER_CLUSTERING.json`
  - Lower-layer summary under `lower_layer_clustering/LOWER_LAYER_CLUSTERING.json`
  - Per-cluster Final-layer artifacts under `final_layer_clustering/final_cluster_<id>/` including `manifest.json`, `assignments.json`, and `cluster_summary.json`
  - Final-layer summary under `final_layer_clustering/FINAL_LAYER_CLUSTERS.json`

## Release Gates

- **Gate G1:** No critical correctness regressions.
- **Gate G2:** No durability/replay regressions.
- **Gate G3:** Contract compatibility maintained for published keys.
- **Gate G4:** Performance not worse than agreed budget for target hardware.
- **Gate G5:** Required stages prove hardware compliance telemetry (`compliance_status=pass`) for CUDA/Tensor Core/Ampere/C++-first policy.
- **Gate G6:** Build or pipeline fails when critical stages silently drop to non-compliant execution.
- **Gate G7:** Required terminal stage-trace events are complete, ordered, and timing-valid.

## Hardware Compliance Test Expectations

- Verify required stages execute with CUDA-enabled critical path.
- Verify Tensor Core activation for eligible INT8/FP16 kernels.
- Verify runtime architecture classification reports Ampere-class target in M1 environments.
- Verify hot-path language/backend telemetry indicates C++/CUDA execution for performance-critical kernels.
- Verify non-compliance raises explicit fail-fast outcome with machine-readable reason and stage ID.
- Validate Ampere-target performance regression thresholds against baseline budgets.

## Terminal Stage Trace Test Expectations

- Fail if any required stage (`Top`, `Mid`, `Lower`, `Final`) is missing `stage_start`.
- Fail if any started stage lacks matching `stage_end` or `stage_fail`.
- Fail if required timing fields are missing or invalid (`elapsed_ms`, `pipeline_elapsed_ms`, timestamps).
- Fail if `pipeline_elapsed_ms` is non-monotonic.
- Fail if event ordering is inconsistent with strict stage dependencies.
- Fail if Lower-layer per-centroid gate/job timing coverage is missing for required centroid jobs.
- Fail if required per-cluster Final-layer timing events are missing for eligible centroid jobs.
- Fail if `stage_fail` omits error fields (`error_code`, `error_message`) or timing-until-failure.

## Final-Layer Per-Cluster Test Expectations

- Verify Final layer starts only after all required Lower-layer gate evaluations and eligible per-centroid jobs are complete.
- Verify Final-layer processing executes independently per eligible gate-fail Lower-layer centroid dataset.
- Fail if Final layer runs for centroid branches with Lower-layer gate decision `continue`.
- Fail if Final layer skips centroid branches with Lower-layer gate decision `stop`.
- Fail if Final-layer processing contains cross-centroid embedding mixing.
- Verify per-cluster Final-layer output artifacts (`manifest.json`, `assignments.json`, `cluster_summary.json`) exist for each eligible centroid dataset.
- Verify aggregate Final-layer summary exists and reconciles with per-cluster outputs.

## Final-Layer `assignments.json` Contract Test Expectations

- Fail if `assignments.json` top-level structure is not an array of objects.
- Fail if any row is missing required fields (`embedding_id`, `final_cluster_id`).
- Fail if rows are not sorted by `embedding_id` ascending.
- Fail if duplicate `embedding_id` rows are present, or expected IDs are missing.
- Fail if `assignments.json` row count does not match embeddings processed for the final cluster.

## Lower-Layer Split-Gate Test Expectations

- Continue split when at least one meaningful subgroup is clearly separated, compact enough, and large enough.
- Stop split when detected subgroup is too small or too noisy.
- Stop split when subgroup separation is not unusual against the local sibling baseline.
- Validate robust-baseline path by enabling robust baseline mode and confirming decision stability.
- Enforce and verify the critical rule: when gate passes, the next split runs on the full parent centroid dataset, not only outliers.

## Decisions and Rationale

- **Decision D-TEST-001:** Keep pipeline script as required gate, not optional.
  - Why: it catches integration failures unit tests miss.
  - Rejected alternative: unit/integration only. Rejected due to low confidence in deployment-like flows.

- **Decision D-TEST-002:** Treat artifact schema checks as first-class tests.
  - Why: scripts and operators depend on these outputs.
  - Rejected alternative: file-existence-only checks. Rejected due to silent schema drift risk.

## Open Questions


## Exit Criteria

- Every requirement in [01 Product and Requirements](./01_PRODUCT_AND_REQUIREMENTS.md) maps to at least one automated test.
- CI gate list is complete and executable on target environment.
