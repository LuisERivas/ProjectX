# 05 Clustering Pipeline

## Purpose

Define v2 clustering design for a 4-layer hierarchy (Top, Mid, Lower, Final/DBSCAN), including artifact contracts and runtime policy.

## Inputs/Dependencies

- [01 Product and Requirements](./01_PRODUCT_AND_REQUIREMENTS.md)
- [03 Storage and Durability](./03_STORAGE_AND_DURABILITY.md)
- [06 API and CLI Contracts](./06_API_AND_CLI_CONTRACTS.md)

## Top Layer Pipeline

1. Load live vectors and packed row-major buffer.
2. Estimate intrinsic dimensionality and derive `k_min`/`k_max`.
3. Run binary elbow selection over integer k-range.
4. Run optional validation stage (stability/quality policy for current state).
5. Persist and atomically replace current artifacts under `clusters/current/`.
6. Update `cluster_manifest.json` for the active clustering state.
7. Emit terminal stage lifecycle events for start and completion/failure, including stage elapsed and cumulative pipeline elapsed.

## Mid Layer Pipeline

1. Start only after Top layer completion is successful.
2. Read Top layer `assignments.json`.
3. Assign each embedding to its top-1 Top-layer centroid.
4. Build one child dataset per centroid using only that centroid's assigned embeddings.
5. Run Mid-layer clustering once globally over these child datasets (single pass, non-recursive).
6. Write Mid-layer artifacts and summary.
7. Emit terminal stage lifecycle events for start and completion/failure, including stage elapsed and cumulative pipeline elapsed.

## Lower Layer Pipeline

1. Start only after Mid layer outputs are available.
2. Continue per-centroid clustering from Mid-layer outputs.
3. Apply a per-centroid continued-processing split gate before each centroid job.
4. If the gate passes, re-split the full parent centroid dataset (outlier subgroup evidence is not the split dataset).
5. For each centroid that passes the gate, recluster its child dataset independently (no cross-centroid mixing).
6. Write centroid-level Lower-layer artifacts and aggregate Lower-layer summary.
7. Emit terminal stage lifecycle events for start and completion/failure, plus per-centroid gate/job timing events.

## Lower-Layer Continued-Processing Split Gate

The split gate is a procedural policy that decides whether a Lower-layer centroid branch should continue splitting.
It runs for each candidate centroid job at the current Lower-layer depth.

### Gate Evidence Inputs

For each candidate parent centroid dataset:

1. Detect an internal outlier set from the top-distance tail relative to the parent centroid.
2. Form candidate outlier subgroups from that outlier set.
3. Build a local sibling-centroid baseline from nearby centroids at the same depth.
4. For each subgroup candidate:
   - evaluate whether separation is unusually large versus the local sibling baseline,
   - evaluate whether subgroup compactness is acceptable relative to the parent,
   - evaluate whether subgroup mass/size is large enough to matter.

### Gate Decision Rule

- Continue splitting only when at least one subgroup passes all required checks:
  - unusually separated versus local sibling baseline,
  - compact enough,
  - large enough.
- If no subgroup passes, stop processing that branch and mark the centroid as a leaf at that depth.

### Critical Processing Rule

- Outlier subgroup structure is evidence only.
- A pass decision authorizes another split over the entire parent centroid dataset, not just the outlier subset.

### Configurable Policy Inputs (M1 Defaults)

- `local_sibling_count` (default: `8`)
- `outlier_percentile` (default: `95`)
- `separation_mode` (default: `standard`, optional: `robust`)
- `separation_threshold` (default: `high`)
- `compactness_threshold` (default: `moderate`)
- `minimum_subgroup_size` (default: `max(32, 1% of parent)`)
- Optional operational guardrails:
  - `max_lower_layer_depth` (default: `3`)
  - `max_centroid_jobs_per_depth` (default: `unbounded` unless resource policy sets a cap)

## Per-Centroid Gate Diagnostics

Required telemetry for each evaluated Lower-layer centroid:

- evaluated centroid ID
- parent dataset size and outlier count
- subgroup count and subgroup sizes
- local sibling baseline summary and baseline mode (`standard` or `robust`)
- per-subgroup pass/fail reason
- final gate decision (`continue` or `stop`)
- if `continue`, explicit marker that next split used full parent dataset
- per-centroid gate evaluation elapsed duration
- per-centroid Lower-layer job elapsed duration when reclustering runs

## Final-Layer Eligibility (Canonical, M1)

This section is the canonical source of Final-layer DBSCAN eligibility semantics for M1.

A Lower-layer centroid dataset is eligible for Final-layer DBSCAN only when all conditions below are true:

1. The centroid branch reached a Lower-layer continued-processing split gate evaluation.
2. The final gate decision for that branch is `stop` (gate check failed).
3. The centroid dataset passes Final-layer DBSCAN preflight validity checks.

Interpretation rules:

- Gate decision `continue` means the branch keeps splitting in Lower layer and is not yet eligible for Final layer.
- Gate decision `stop` means the branch is a gate-fail leaf candidate and becomes eligible for Final-layer DBSCAN only if DBSCAN preflight checks pass.

## Final-Layer DBSCAN Preflight Validity Rules (M1 Required)

A centroid dataset is DBSCAN-valid in M1 only if all checks below pass:

1. Dataset is non-empty.
2. Point count meets minimum policy threshold (`n_points >= min_points_policy`).
3. Vectors have consistent dimensionality and match configured dimension.
4. All numeric values are finite (no `NaN`, no `Inf`).
5. ID/vector alignment integrity checks pass.

Failure behavior (required):

- DBSCAN must not run for centroid datasets that fail preflight checks.
- A machine-readable preflight failure/skip reason code must be emitted in terminal event output.
- The same reason code must be persisted in per-centroid manifest/summary telemetry.

## Final Layer Pipeline (DBSCAN)

1. Start only after all required Lower-layer gate evaluations and eligible per-centroid jobs are complete.
2. Iterate only over eligible gate-fail leaf centroid datasets defined in `Final-Layer Eligibility (Canonical, M1)`.
3. For each eligible centroid dataset, run required DBSCAN preflight validity checks.
4. Run DBSCAN independently only for datasets that pass preflight checks.
5. Enforce no cross-centroid mixing during Final-layer DBSCAN.
6. Write required per-centroid Final-layer artifacts (`manifest.json`, `labels.json`, `cluster_summary.json`) for datasets that pass preflight checks.
7. Write aggregate Final-layer summary.
8. Emit terminal stage lifecycle events for start and completion/failure, including stage elapsed and cumulative pipeline elapsed.
9. Emit per-centroid Final-layer timing events (centroid/job timing and status), including explicit preflight outcome and reason code.

## Terminal Stage Trace Requirement (M1)

Terminal stage-by-stage progress and timing output is mandatory by default for all pipeline executions (not debug-only).
All required stage and substage events must be emitted in execution order and be machine-parseable.

Required trace coverage:

- Top Layer stage lifecycle events
- Mid Layer stage lifecycle events
- Lower Layer stage lifecycle events
- Final Layer (DBSCAN) stage lifecycle events
- Required per-centroid Lower-layer gate/job timing events
- Required per-centroid Final-layer DBSCAN timing events
- Pipeline summary event with cumulative elapsed timing

## Artifact Contract (Required)

- `id_estimate.json`
- `elbow_trace.json`
- `centroids.bin`
- `assignments.json`
- `stability_report.json` (or policy-equivalent validation report)
- `cluster_manifest.json`
- `mid_layer_clustering/MID_LAYER_CLUSTERING.json`
- `lower_layer_clustering/LOWER_LAYER_CLUSTERING.json`
- `final_layer_clustering/FINAL_LAYER_DBSCAN.json`
- `final_layer_clustering/centroid_<id>/manifest.json`
- `final_layer_clustering/centroid_<id>/labels.json`
- `final_layer_clustering/centroid_<id>/cluster_summary.json`

### `labels.json` Contract (Final-Layer Per-Centroid, M1 Required)

- Top-level structure must be an array of objects.
- Each object must include:
  - `embedding_id` (same ID type used by system contracts)
  - `label` (integer DBSCAN cluster label)
- Noise label convention is required: `label = -1` means DBSCAN noise.
- Entries must be sorted by `embedding_id` ascending.
- Each `embedding_id` must appear exactly once.
- Row count must equal the number of embeddings processed for that centroid dataset.

## Runtime and Precision Policy

- CUDA-required execution for performance-critical scoring, k-selection, clustering, and assignment kernels across Top/Mid/Lower stages.
- Tensor Core utilization is a primary execution target for eligible INT8/FP16 kernels, not just a telemetry note.
- M1 optimization target is Ampere-class GPU runtime behavior for eligible kernels and stage orchestration.
- M1 numeric policy: embeddings are received/stored as FP32; optimal cluster-count estimation (k-selection) is computed in INT8; final clustering and assignment after k is selected are computed in FP16.
- INT8 applies only to k-selection stages (ID estimate / elbow / k-search logic).
- FP16 applies to non-DBSCAN clustering and assignment stages after `k` is selected.
- Implementation policy: performance-critical hot paths are C++-first (C++/CUDA); higher-level scripting does not replace core compute kernels.
- Final-layer DBSCAN stage follows the same C++/CUDA implementation target for performance-critical execution; any exception must be explicit and telemetry-visible.
- Fail-fast policy: if required CUDA/Ampere/Tensor Core compliance is not met in performance-critical stage execution, the stage fails with explicit machine-readable non-compliance reason.

## Decisions and Rationale

- **Decision D-CLUST-001:** Preserve hierarchical Top/Mid/Lower/Final clustering with one active artifact set in M1.
  - Why: this supports hierarchical structure discovery and inspection.
  - Rejected alternative: single global layer only. Rejected due to lower representational granularity for heterogeneous datasets.

- **Decision D-CLUST-002:** Keep explicit JSON artifact contract for each stage.
  - Why: needed for debugging, auditability, and script automation.
  - Rejected alternative: binary-only opaque outputs. Rejected due to weak operability.
- **Decision D-CLUST-003:** Treat Ampere/CUDA/Tensor Core and C++-first hot-path execution as mandatory M1 compliance constraints.
  - Why: performance-critical behavior is a core product requirement, not an optional optimization.
  - Rejected alternative: soft best-effort GPU usage with permissive CPU fallback. Rejected due to silent performance regressions and non-deterministic operational behavior.

## Open Questions

## Exit Criteria

- All clustering outputs are deterministic and script-consumable.
- End-to-end runner can print Top/Mid/Lower/Final summaries without hidden assumptions.
