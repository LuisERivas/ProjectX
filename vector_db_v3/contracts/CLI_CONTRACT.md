# CLI Contract

## Purpose

Define stable CLI commands, output behavior, and failure semantics for M1.

## Required Commands (M1)

- `init --path <data_dir>`
- `insert --path ... --id <u64> --vec <file_or_csv>`
- `bulk-insert --path ... --input <jsonl> [--batch-size <u32>]`
- `bulk-insert-bin --path ... --input <bin> [--batch-size <u32>]`
- `delete --path ... --id <u64>`
- `get --path ... --id <u64>`
- `search --path ... --vec <file_or_csv> [--topk <u32>]`
- `stats --path ...`
- `wal-stats --path ...`
- `checkpoint --path ...`
- `build-top-clusters --path ... [--seed <u32>]`
- `build-mid-layer-clusters --path ... [--seed <u32>]`
- `build-lower-layer-clusters --path ... [--seed <u32>]`
- `build-final-layer-clusters --path ... [--seed <u32>]`
- `cluster-stats --path ...`
- `cluster-health --path ...`

## Exit Code Contract

- `0`: command completed successfully.
- `1`: command failed (validation/runtime/contract/compliance error).
- `2`: command usage error (missing/invalid required args).

## Stdout/Stderr Contract

- Machine-readable command outputs are emitted to stdout.
- Human-readable errors are emitted to stderr in `error: <message>` form.
- Terminal lifecycle events are emitted as JSONL to stdout for pipeline stages.

## Query Contract (M1)

- Search is exact-only.
- No metadata filtering.
- No metadata-influenced ranking.
- Result rows must be sorted by score descending, then `embedding_id` ascending tie-break.


## Backward Compatibility

- Existing output keys must not silently change meaning.
- New keys must be additive.
- Breaking changes require explicit migration note and version bump.

## Binary Ingest Feed (M1 additive)

- `bulk-insert-bin` reads a little-endian FP32 feed with fixed records.
- Header layout:
  - `magic: u32` (`V3BI`)
  - `version: u16` (`1`)
  - `record_size_bytes: u32` (`4104`)
  - `record_count: u64`
- Record layout:
  - `embedding_id: u64`
  - `vector: 1024 x f32` (FP32 ingest boundary)

