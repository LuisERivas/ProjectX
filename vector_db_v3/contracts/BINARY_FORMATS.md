# Binary Formats Contract

## Purpose

Define canonical byte-level layouts for all `.bin` artifacts referenced by `ARTIFACT_CONTRACT.md`.

## Global Rules

- Endianness: little-endian.
- Integer widths: fixed (`u8`, `u16`, `u32`, `u64`).
- Float widths: IEEE-754 (`f32`).
- Record order: row-major, contiguous, no padding between records unless explicitly stated.
- Unless otherwise noted, all `reserved` fields must be zero.
- All binary files must be written atomically (temp-write then rename).
- No quantization sidecar metadata binaries are required in M1.
- Precision artifact consistency is canonicalized by `embedding_id` alignment across FP32/FP16/INT8 artifacts.

## Common Header (For Summary/Manifest Binary Artifacts)

Used by:

- `cluster_manifest.bin`
- `MID_LAYER_CLUSTERING.bin`
- `LOWER_LAYER_CLUSTERING.bin`
- `FINAL_LAYER_CLUSTERS.bin`
- `final_cluster_<id>/manifest.bin`
- `final_cluster_<id>/cluster_summary.bin`

Layout (`16` bytes total):

| Offset | Size | Type | Field |
|---|---:|---|---|
| 0 | 2 | `u16` | `schema_version` |
| 2 | 2 | `u16` | `record_type` |
| 4 | 4 | `u32` | `record_count` |
| 8 | 4 | `u32` | `payload_bytes` |
| 12 | 4 | `u32` | `checksum_crc32` |

## `id_estimate.bin`

Layout (`12` bytes total):

| Offset | Size | Type | Field |
|---|---:|---|---|
| 0 | 4 | `u32` | `k_min` |
| 4 | 4 | `u32` | `k_max` |
| 8 | 2 | `u16` | `id_estimate_method` |
| 10 | 2 | `u16` | `reserved` |

Invariants:

- `k_min > 0`
- `k_max > 0`
- `k_min <= k_max`

## `elbow_trace.bin`

Per-record layout (`12` bytes per row):

| Offset | Size | Type | Field |
|---|---:|---|---|
| 0 | 4 | `u32` | `k_value` |
| 4 | 4 | `f32` | `objective_value` |
| 8 | 1 | `u8` | `probe_phase` |
| 9 | 3 | `u8[3]` | `reserved` |

`probe_phase` enum:

- `1` = coarse
- `2` = fine

Invariants:

- At least one record where `k_value == chosen_k` for that stage run.

## `stability_report.bin`

Layout (`20` bytes total):

| Offset | Size | Type | Field |
|---|---:|---|---|
| 0 | 2 | `u16` | `status_code` |
| 2 | 2 | `u16` | `reserved` |
| 4 | 4 | `f32` | `mean_nmi` |
| 8 | 4 | `f32` | `std_nmi` |
| 12 | 4 | `f32` | `mean_jaccard` |
| 16 | 4 | `f32` | `mean_centroid_drift` |

`status_code` enum:

- `0` = unknown
- `1` = ok
- `2` = warning
- `3` = fail

## `assignments.bin` (Top)

Per-record layout (`12` bytes per row):

| Offset | Size | Type | Field |
|---|---:|---|---|
| 0 | 8 | `u64` | `embedding_id` |
| 8 | 4 | `u32` | `top_centroid_numeric_id` |

## `mid_layer_clustering/assignments.bin` (Mid)

Per-record layout (`16` bytes per row):

| Offset | Size | Type | Field |
|---|---:|---|---|
| 0 | 8 | `u64` | `embedding_id` |
| 8 | 4 | `u32` | `mid_centroid_numeric_id` |
| 12 | 4 | `u32` | `parent_top_centroid_numeric_id` |

## `final_cluster_<id>/assignments.bin` (Final)

Per-record layout (`12` bytes per row):

| Offset | Size | Type | Field |
|---|---:|---|---|
| 0 | 8 | `u64` | `embedding_id` |
| 8 | 4 | `u32` | `final_cluster_numeric_id` |

Assignment invariants (Top/Mid/Final):

- Sorted by `embedding_id` ascending.
- No duplicate `embedding_id` in artifact domain.
- Record count equals processed row count in corresponding stage metadata.

## `k_search_bounds_batch.bin`

Per-record layout (`24` bytes per row):

| Offset | Size | Type | Field |
|---|---:|---|---|
| 0 | 1 | `u8` | `stage_level` |
| 1 | 1 | `u8` | `gate_decision` |
| 2 | 2 | `u16` | `reserved` |
| 4 | 4 | `u32` | `source_numeric_id` |
| 8 | 4 | `u32` | `k_min` |
| 12 | 4 | `u32` | `k_max` |
| 16 | 4 | `u32` | `chosen_k` |
| 20 | 4 | `u32` | `dataset_size` |

Enums:

- `stage_level`: `1=Top`, `2=Mid`, `3=Lower`
- `gate_decision`: `0=not_applicable`, `1=continue`, `2=stop`

Invariants:

- Sorted by `(stage_level, source_numeric_id)` ascending.
- `k_min <= chosen_k <= k_max`.
- `stage_level=1` or `2` must use `gate_decision=0`.
- `stage_level=3` must use `gate_decision` in `{1,2}`.

## `post_cluster_membership.bin`

Per-record layout (`24` bytes per row):

| Offset | Size | Type | Field |
|---|---:|---|---|
| 0 | 8 | `u64` | `embedding_id` |
| 8 | 4 | `u32` | `top_centroid_numeric_id` |
| 12 | 4 | `u32` | `mid_centroid_numeric_id` |
| 16 | 4 | `u32` | `lower_centroid_numeric_id` |
| 20 | 4 | `u32` | `final_cluster_numeric_id` |

Invariants:

- Sorted by `embedding_id` ascending.
- No duplicate `embedding_id`.
- Exactly one row per live embedding in active clustering state.
- ID fields must resolve via corresponding stage/final mapping tables.
- If lower-level mapping is unavailable by design, `lower_centroid_numeric_id` uses sentinel `UINT32_MAX`.

## ID Mapping Rule

Numeric centroid/cluster IDs in binary files must resolve via the corresponding stage/final manifest mapping tables.

## Precision ID-Alignment Rule (M1 Canonical)

For active-state precision artifacts (`embeddings_fp32.bin`, `embeddings_fp16.bin` when present, `embeddings_int8*.bin` when present):

- Rows must be sorted by `embedding_id` ascending.
- `embedding_id` values must be unique per file.
- `embedding_id` cardinality must match across precision variants.
- `embedding_id` membership set must match exactly across precision variants.
- Optional strict mode may additionally require row-index alignment across precision files.

Quantization sidecar metadata policy:

- Sidecar metadata formats such as `embeddings_int8_*.meta.bin` are not required in M1.
- Binary row layouts remain unchanged; consistency validation is ID-alignment-based.

## Compatibility and Versioning

- `schema_version` bump is required for any binary layout change.
- Readers must reject unknown mandatory schema versions unless compatibility logic is explicitly implemented.

