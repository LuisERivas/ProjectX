# Artifact Contract

## Purpose

Define required artifacts, per-stage output rules, and schema invariants for M1.
Canonical byte-level layouts are specified in `BINARY_FORMATS.md`.

## Encoding Policy (M1)

- Binary-first is mandatory for large data-plane artifacts.
- JSON is allowed only where human-readable control-plane metadata is explicitly required.
- If both binary and JSON representations exist, binary is canonical and JSON is derivative/debug-only.
- Stage-level artifacts in this contract are binary unless explicitly marked JSON.
- Quantization metadata sidecar artifacts are not required in M1.
- Precision consistency is enforced by `embedding_id` alignment across precision artifacts.

## Root Layout (M1)

- `<data_dir>/manifest.json`
- `<data_dir>/wal.log`
- `<data_dir>/segments/...`
- `<data_dir>/clusters/current/...`

## Required Clustering Artifacts

Under `clusters/current/`:

- `id_estimate.bin` (canonical)
- `elbow_trace.bin` (canonical)
- `centroids.bin`
- `assignments.bin` (Top-layer assignments; canonical)
- `stability_report.bin` (canonical)
- `cluster_manifest.bin` (canonical stage manifest)
- `mid_layer_clustering/MID_LAYER_CLUSTERING.bin` (canonical)
- `mid_layer_clustering/assignments.bin` (canonical)
- `lower_layer_clustering/LOWER_LAYER_CLUSTERING.bin` (canonical)
- `final_layer_clustering/FINAL_LAYER_CLUSTERS.bin` (canonical)
- `final_layer_clustering/final_cluster_<id>/manifest.bin` (canonical)
- `final_layer_clustering/final_cluster_<id>/assignments.bin` (canonical)
- `final_layer_clustering/final_cluster_<id>/cluster_summary.bin` (canonical)
- `k_search_bounds_batch.bin` (canonical end-of-pipeline batch artifact)
- `post_cluster_membership.bin` (canonical end-of-pipeline consolidated membership artifact)

## Top/Mid/Final Assignment Binary Contract

Applicable files:

- `clusters/current/assignments.bin`
- `clusters/current/mid_layer_clustering/assignments.bin`
- `clusters/current/final_layer_clustering/final_cluster_<id>/assignments.bin`

Required binary layout (little-endian, row-major records):

- Top assignments row:
  - `embedding_id` (`u64`)
  - `top_centroid_numeric_id` (`u32`)
- Mid assignments row:
  - `embedding_id` (`u64`)
  - `mid_centroid_numeric_id` (`u32`)
  - `parent_top_centroid_numeric_id` (`u32`)
- Final assignments row:
  - `embedding_id` (`u64`)
  - `final_cluster_numeric_id` (`u32`)

Naming/string IDs for centroids/clusters must be declared in the matching `manifest.bin` mapping table.

Row-level invariants:

- Rows are sorted by `embedding_id` ascending.
- `embedding_id` appears exactly once within each assignment artifact domain.
- Row count equals number of embeddings processed for that artifact.

## `id_estimate.bin` Contract

Applicable file:

- `clusters/current/id_estimate.bin`

Required binary layout (little-endian, fixed record):

- `k_min` (`u32`)
- `k_max` (`u32`)
- `id_estimate_method` (`u16`)  // implementation-defined enum
- `reserved` (`u16`)            // set to 0

Invariants:

- Exactly one record for each stage-level run that emits id-estimate output.
- Values must be non-zero and satisfy `k_min <= k_max`.

## `elbow_trace.bin` Contract

Applicable file:

- `clusters/current/elbow_trace.bin`

Required binary layout (little-endian, row-major records):

- `k_value` (`u32`)
- `objective_value` (`f32`)
- `probe_phase` (`u8`)   // 1=coarse, 2=fine
- `reserved` (`u8[3]`)   // set to 0

Invariants:

- Contains all evaluated k-probe points for the stage run.
- Includes at least one record where `k_value == chosen_k`.

## `stability_report.bin` Contract

Applicable file:

- `clusters/current/stability_report.bin`

Required binary layout (little-endian, fixed record):

- `status_code` (`u16`)   // implementation-defined enum
- `reserved` (`u16`)      // set to 0
- `mean_nmi` (`f32`)
- `std_nmi` (`f32`)
- `mean_jaccard` (`f32`)
- `mean_centroid_drift` (`f32`)

## End-of-Pipeline K-Search Bounds Batch Contract

Applicable file:

- `clusters/current/k_search_bounds_batch.bin`

Purpose:

- Persist all k-search bounds across clustering stages in one large batch written at the end of pipeline processing.

Write timing rule:

- This artifact is written in a dedicated final step that runs after Final layer completion.

Required binary layout (little-endian, row-major records):

- `stage_level` (`u8`)          // 1=Top, 2=Mid, 3=Lower
- `gate_decision` (`u8`)        // 0=not_applicable, 1=continue, 2=stop
- `reserved` (`u16`)            // set to 0
- `source_numeric_id` (`u32`)   // stage-scoped numeric centroid/source id
- `k_min` (`u32`)
- `k_max` (`u32`)
- `chosen_k` (`u32`)
- `dataset_size` (`u32`)

Row-level invariants:

- Contains one row per k-search execution performed in Top/Mid/Lower stages.
- Rows are sorted by `(stage_level, source_numeric_id)` ascending.
- `k_min <= chosen_k <= k_max` for every row.
- If `stage_level=1` (Top), `gate_decision` must be `0`.
- If `stage_level=2` (Mid), `gate_decision` must be `0`.
- If `stage_level=3` (Lower), `gate_decision` must be `1` or `2`.

## End-of-Pipeline Post-Cluster Membership Contract

Applicable file:

- `clusters/current/post_cluster_membership.bin`

Purpose:

- Persist one consolidated membership row per embedding for fast downstream lookup without joining stage-specific assignment artifacts.

Write timing rule:

- This artifact is written in a dedicated finalization step that runs after Final layer completion.

Required binary layout (little-endian, row-major records):

- `embedding_id` (`u64`)
- `top_centroid_numeric_id` (`u32`)
- `mid_centroid_numeric_id` (`u32`)
- `lower_centroid_numeric_id` (`u32`)
- `final_cluster_numeric_id` (`u32`)

Row-level invariants:

- Contains one row per live embedding in the active clustering state.
- Rows are sorted by `embedding_id` ascending.
- `embedding_id` appears exactly once.
- `top_centroid_numeric_id`, `mid_centroid_numeric_id`, and `final_cluster_numeric_id` must resolve via stage/final manifest mapping tables.
- `lower_centroid_numeric_id` must resolve via lower-layer mapping when available; if unavailable by design, use sentinel `UINT32_MAX`.

## Precision Artifact Consistency Contract (ID-Alignment Canonical)

Applicable precision artifact set for active state:

- `embeddings_fp32.bin` (canonical source)
- `embeddings_fp16.bin` (when present)
- `embeddings_int8*.bin` (when present)

Quantization metadata sidecars are not required:

- `embeddings_int8_*.meta.bin` or equivalent sidecar formats are not required in M1.
- Scale/zero-point/quantization-mode sidecar metadata files are not contract requirements.

Required ID-alignment rules:

- `embedding_id` must be unique per precision file.
- Rows must be sorted by `embedding_id` ascending per precision file.
- `embedding_id` cardinality must match across precision variants in the active state.
- `embedding_id` membership set must match exactly across precision variants in the active state.
- Optional strict mode: row-index alignment across precision files may be required by deployment policy.

Invalidation and runtime behavior:

- If any required alignment rule fails, the precision artifact set is invalid.
- Runtime must deterministically either:
  - regenerate derived precision artifacts from canonical source, or
  - fail the stage with explicit machine-readable error.
- Stage success with failed alignment and without regeneration is non-compliant.

## Stage Summary and Manifest Binary Contracts

Applicable files:

- `clusters/current/cluster_manifest.bin`
- `clusters/current/mid_layer_clustering/MID_LAYER_CLUSTERING.bin`
- `clusters/current/lower_layer_clustering/LOWER_LAYER_CLUSTERING.bin`
- `clusters/current/final_layer_clustering/FINAL_LAYER_CLUSTERS.bin`
- `clusters/current/final_layer_clustering/final_cluster_<id>/manifest.bin`
- `clusters/current/final_layer_clustering/final_cluster_<id>/cluster_summary.bin`

Required common header fields for all summary/manifest binary artifacts:

- `schema_version` (`u16`)
- `record_type` (`u16`)   // implementation-defined enum
- `record_count` (`u32`)
- `payload_bytes` (`u32`)
- `checksum_crc32` (`u32`)

## Required Metadata Fields For Binary Artifacts

Every binary summary/manifest payload must encode:

- `artifact_path`
- `artifact_format` (for example `assignments.bin.v1`)
- `endianness` (`little`)
- `record_size_bytes`
- `record_count`
- `schema_version`
- `checksum` (recommended `sha256`)
- optional ID mapping dictionary for numeric-to-string centroid identifiers

For `k_search_bounds_batch.bin`, manifest metadata must additionally include:

- `rows_top`
- `rows_mid`
- `rows_lower`
- `pipeline_step_name` (expected: `finalize_k_search_bounds_batch`)

Root store manifest exception:

- `<data_dir>/manifest.json` remains JSON for root-level bootstrap/recovery compatibility in M1.

## Integrity Rules

- All required artifact files must exist for a successful stage.
- Aggregate summaries must reconcile with per-cluster/per-centroid outputs.
- Missing required artifact implies stage failure.

## Atomicity Rules

- Artifact writes must use temp-write then atomic replace.
- No partial file should be visible as final output.

## Versioning Rule (M1)

- M1 keeps one active artifact state.
- Artifact history/version retention is deferred.

