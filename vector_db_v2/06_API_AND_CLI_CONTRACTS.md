# 06 API and CLI Contracts

## Purpose

Define stable command/response contracts so scripts and tools can evolve without frequent breakage.

## Inputs/Dependencies

- [02 System Architecture](./02_SYSTEM_ARCHITECTURE.md)
- [03 Storage and Durability](./03_STORAGE_AND_DURABILITY.md)
- [05 Clustering Pipeline](./05_CLUSTERING_PIPELINE.md)

## CLI Command Surface (M1)

- `init --path <data_dir>`
- `insert --path ...`
- `bulk-insert --path ... --input <jsonl>`
- `delete --path ... --id <u64>`
- `get --path ... --id <u64>`
- `stats --path ...`
- `wal-stats --path ...`
- `checkpoint --path ...`
- `build-top-clusters --path ... [--seed <u32>]`
- `build-mid-layer-clusters --path ... [--seed <u32>]`
- `build-lower-layer-clusters --path ... [--seed <u32>]`
- `build-final-layer-clusters --path ... [--seed <u32>]`
- `cluster-stats --path ...`
- `cluster-health --path ...`

Deferred post-M1:

- `update-meta --path ... --id <u64> --meta <json_patch>` (deferred because M1 record model is minimal `{embedding_id, vector}`)

## Query Contract (M1)

- Query behavior is exact vector-similarity ranking only.
- Query response shape is minimal and ranking-oriented:
  - `embedding_id`
  - `score`
- M1 does not support metadata-based filtering or metadata-influenced ranking.

## Clustering Stage Contract (M1)

- Stage order is strict: Top -> Mid -> Lower -> Final.
- Mid starts only after Top succeeds.
- Lower starts only after Mid outputs are available.
- Final starts only after all required Lower-layer gate evaluations and eligible per-centroid jobs complete.
- Final-layer eligibility follows `05` canonical rule: gate decision must be `stop`, and centroid dataset must pass required DBSCAN preflight validity checks.
- Gate decision `continue` is not Final-layer eligible and remains in Lower-layer processing.
- Final layer runs DBSCAN per eligible gate-fail Lower-layer centroid dataset.
- Final layer must not mix embeddings across centroid datasets.
- Final layer produces both per-centroid outputs and an aggregate summary.

## Final-Layer Per-Centroid Artifact Contract (M1 Minimal Normative)

For each eligible centroid dataset under `final_layer_clustering/centroid_<id>/`, required files are:

- `manifest.json`
- `labels.json`
- `cluster_summary.json`

`labels.json` required schema constraints:

- Top-level JSON value is an array of objects.
- Each object contains required fields: `embedding_id` and `label`.
- `label` is an integer; `-1` is required DBSCAN noise convention.
- Entries are sorted by `embedding_id` ascending.
- `embedding_id` values are unique (exactly one entry per embedding).
- Entry count equals embeddings processed for that centroid dataset.

Optional files may be added without breaking M1 contracts.
Aggregate summary remains required at `final_layer_clustering/FINAL_LAYER_DBSCAN.json`.

## Final-Layer Per-Centroid Result Reporting (M1)

Per-centroid Final-layer result reporting must expose:

- preflight validity outcome (`pass` or `fail`)
- preflight failure/skip reason code (when outcome is `fail`)
- `labels.json` presence status
- `labels.json` schema validity status
- per-centroid output status (`written`, `skipped_preflight_failed`, `failed`)

## Contract Rules

- JSON responses are machine-readable and stable for existing keys.
- New fields are additive and optional for older clients.
- Existing field meaning cannot silently change.
- Breaking changes require explicit contract migration notes.

## Required `cluster-stats` Keys (M1 Baseline)

- `available`, `build_lsn`
- `vectors_indexed`, `chosen_k`, `k_min`, `k_max`, `objective`
- `used_cuda`, `tensor_core_enabled`, `gpu_backend`
- timing + telemetry keys used by operational scripts
- fields above describe the current active clustering state in M1

## Hardware Compliance Telemetry (M1 Required)

Machine-verifiable fields must prove execution quality for performance-critical stages:

- `cuda_required` (boolean policy flag for critical path)
- `cuda_enabled` (boolean runtime flag)
- `tensor_core_required` (boolean policy flag for eligible kernels)
- `tensor_core_active` (boolean runtime result for eligible kernels)
- `gpu_arch_class` (for example: `ampere`)
- `kernel_backend_path` (for example: `cuda_cublaslt`, `cuda_custom_kernel`)
- `hot_path_language` (expected: `cpp_cuda`)
- `compliance_status` (`pass` or `fail`)
- `fallback_reason` (non-empty only when non-compliant path is detected)
- `non_compliance_stage` (stage identifier when `compliance_status=fail`)

## Compliance Semantics (M1)

- CI and validation pipelines must fail if `compliance_status=fail` for required stages.
- Silent downgrade to non-compliant execution in critical path is disallowed.
- Diagnostic/non-critical tooling may use non-compliant paths only when explicitly marked and excluded from critical-path gates.

## Terminal Event Contract (M1)

Terminal stage trace output is required by default for all pipeline executions.
Required format is machine-parseable JSON lines on stdout for stage lifecycle and summary events.

Required event fields:

- `event_type` (`stage_start`, `stage_end`, `stage_fail`, `pipeline_summary`)
- `stage_id`
- `stage_name`
- `status`
- `start_ts`
- `end_ts` (required for `stage_end` and `stage_fail`)
- `elapsed_ms`
- `pipeline_elapsed_ms`
- `records_processed` (when applicable)
- `centroid_id` and/or `job_id` (required for per-centroid Lower-layer and per-centroid Final-layer events when applicable)
- `gate_decision` (`continue` or `stop`) for Lower-layer gate evaluation events
- `final_layer_eligibility_reason` (for example: `gate_stop_and_preflight_pass`, or explicit ineligible reason)
- `final_layer_preflight_valid` (`true` or `false`) for Final-layer per-centroid preflight events
- `preflight_reason_code` (required when `final_layer_preflight_valid=false`)
- `labels_file_present` (`true` or `false`) when per-centroid output status is reported
- `labels_schema_valid` (`true` or `false`) when `labels.json` is present
- `final_layer_output_status` (for example: `written`, `skipped_preflight_failed`, `failed`)
- `error_code` and `error_message` (required for failure events)
- `non_compliance_stage` (required for hardware non-compliance failures)

Ordering and validity rules:

- Events are emitted in execution order.
- Every `stage_start` has exactly one matching `stage_end` or `stage_fail`.
- `elapsed_ms` and `pipeline_elapsed_ms` are non-negative.
- `pipeline_elapsed_ms` is monotonic over event stream.

Stdout/stderr responsibilities:

- Stage lifecycle and summary events are emitted to stdout as machine-parseable JSON lines.
- Human-readable diagnostics may be emitted in parallel.
- Fatal errors must include `stage_fail` with timing-until-failure and explicit error fields; no silent failure.
- Final-layer DBSCAN preflight failures must emit machine-readable skip/failure reason codes; silent preflight skips are disallowed.

## Required `cluster-health` Keys (M1 Baseline)

- `available`, `passed`, `status`
- `mean_nmi`, `std_nmi`, `mean_jaccard`, `mean_centroid_drift`

## Error Contract

- Non-zero process exit for command failure.
- Human-readable `error: <message>` on stderr.
- No partial success reported as success.

## Decisions and Rationale

- **Decision D-CONTRACT-001:** Keep CLI as primary automation contract for M1.
  - Why: existing scripts already rely on CLI commands and JSON outputs.
  - Rejected alternative: immediate RPC-only interface. Rejected due to migration cost and tooling disruption.

- **Decision D-CONTRACT-002:** Prefer additive JSON evolution.
  - Why: minimizes downstream script breakages during feature growth.
  - Rejected alternative: strict schema replacement per release. Rejected due to high operational churn.

## Open Questions

- Should we publish JSON schema files (`.schema.json`) for each command in M1?
  - Decision: Not required in M1.
- Should command aliases be supported for old names in migration windows?
  - Decision: Not required in M1.

## Exit Criteria

- Pipeline scripts can run without brittle parsing assumptions.
- Contract docs are precise enough to implement parser tests in [07 Testing and Validation](./07_TESTING_AND_VALIDATION.md).
