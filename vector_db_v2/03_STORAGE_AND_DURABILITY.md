# 03 Storage and Durability

## Purpose

Define the on-disk data model, WAL/checkpoint behavior, and crash-recovery guarantees for Vector DB v2.

## Inputs/Dependencies

- [01 Product and Requirements](./01_PRODUCT_AND_REQUIREMENTS.md)
- [02 System Architecture](./02_SYSTEM_ARCHITECTURE.md)
- [06 API and CLI Contracts](./06_API_AND_CLI_CONTRACTS.md)

## Proposed Disk Layout (v2)

```text
<data_dir>/
  manifest.json
  wal.log
  dirty_ranges.json
  segments/
    seg_<id>.vec
    seg_<id>.ids
    seg_<id>.tomb
  clusters/
    current/
      cluster_manifest.json
      id_estimate.json
      elbow_trace.json
      stability_report.json
      centroids.bin
      assignments.json
      mid_layer_clustering/
        MID_LAYER_CLUSTERING.json
      lower_layer_clustering/
        LOWER_LAYER_CLUSTERING.json
        centroid_<id>/
          manifest.json
          ...
      final_layer_clustering/
        FINAL_LAYER_DBSCAN.json
        centroid_<id>/
          manifest.json
          labels.json
          cluster_summary.json
          ...
```

## Durability Model

- **Write-ack policy:** an operation is acknowledged only after WAL append succeeds.
- **Checkpoint policy:** flushes current durable state and truncates WAL to bound replay cost.
- **Recovery policy:** on open, replay WAL records after checkpoint LSN to reconstruct latest state.
- **Tombstone policy:** deletes are logical and represented in tomb files/state maps.

## Crash Consistency Guarantees

- If process crashes after WAL append but before checkpoint, data is recovered via replay.
- If process crashes during artifact writing, atomic file writes must prevent partial-file visibility.
- Manifest writes use atomic replace semantics to avoid torn metadata.

## Data Integrity Rules

- `manifest.json` is the source of active segment metadata and checkpoint LSN.
- Segment row mapping consistency:
  - `.ids` index aligns with vector rows in `.vec`
  - tombstone state resolves final live/deleted visibility
- Cluster manifest and clustering artifacts represent one active state in M1 and are atomically replaced on rebuild.
- Final-layer per-centroid artifact directories are created only for eligible gate-fail (`stop`) leaf centroid datasets that pass required DBSCAN preflight checks (non-empty, minimum points policy, dimensional consistency, finite numeric values, and ID/vector alignment integrity; canonical rule in `05`).

## Decisions and Rationale

- **Decision D-STOR-001:** Continue WAL + checkpoint model for M1.
  - Why: simple, testable durability baseline with clear replay semantics.
  - Rejected alternative: append-only segments with no WAL. Rejected due to complicated multi-file transactional guarantees.

- **Decision D-STOR-002:** Keep atomic write-then-rename for manifest/artifact files.
  - Why: prevents partially written JSON/bin visibility under crashes.
  - Rejected alternative: direct overwrite writes. Rejected due to higher corruption risk.

## Open Questions

## Exit Criteria

- Recovery behavior is covered by replay and checkpoint tests.
- File layout and invariants are frozen for M1 implementation.
- Contract fields referenced in [06 API and CLI Contracts](./06_API_AND_CLI_CONTRACTS.md) map directly to persisted state.
