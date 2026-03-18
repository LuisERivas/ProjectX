# `pipeline_test.py` Usage Guide

## Purpose

`pipeline_test.py` is a synthetic pipeline driver for `vector_db_v3` that can:

- generate synthetic 1024-dim embeddings (`jsonl` or `bin`)
- run smoke stages with latency logs
- run the full stage pipeline in order (`init -> ingest -> top -> mid -> lower -> final`)
- optionally run `search` sanity and `cluster-stats`
- parse and print cluster summary counts
- always write a machine-readable results JSON file (`--results-out` or default path)

## Prerequisites

- Python 3.10+ available as `python3` (or `python`)
- built CLI binary at `vector_db_v3/build/vectordb_v3_cli` (or `.exe` on Windows)
- run commands from repository root

Build example:

```bash
cmake -S vector_db_v3 -B vector_db_v3/build
cmake --build vector_db_v3/build -j
```

## Quick Start

Generation-only run (no stage execution):

```bash
python3 vector_db_v3/scripts/pipeline_test.py \
  --build-dir vector_db_v3/build \
  --embedding-count 256 \
  --input-format bin
```

## CLI Arguments

- `--build-dir` (required): directory containing `vectordb_v3_cli(.exe)`
- `--embedding-count` (required): number of embeddings to generate (`> 0`)
- `--data-dir` (optional): working/output directory (default temp path)
- `--batch-size` (optional): ingest batch size, default `1000`, must be `> 0`
- `--input-format` (optional): `bin|jsonl`, default `bin`
- `--seed` (optional): deterministic seed; if omitted, entropy mode is used
- `--results-out` (optional): result JSON path (default `<data_dir>/pipeline_test_results.json`)
- `--keep-data` (optional): preserve generated data directory
- `--runner-smoke` (optional): run smoke stages (`init`, `stats`) with lifecycle logs
- `--run-full-pipeline` (optional): run full ordered stages
- `--with-search-sanity` (optional): append `search` stage; requires `--run-full-pipeline`
- `--with-cluster-stats` (optional): append `cluster-stats` stage; requires `--run-full-pipeline`

Constraints:

- `--runner-smoke` cannot be combined with `--run-full-pipeline`
- `--with-search-sanity` and `--with-cluster-stats` require `--run-full-pipeline`

## Usage Examples

Deterministic run with seed:

```bash
python3 vector_db_v3/scripts/pipeline_test.py \
  --build-dir vector_db_v3/build \
  --data-dir /tmp/v3_seeded \
  --embedding-count 256 \
  --seed 123 \
  --input-format bin
```

Entropy run without seed:

```bash
python3 vector_db_v3/scripts/pipeline_test.py \
  --build-dir vector_db_v3/build \
  --data-dir /tmp/v3_entropy \
  --embedding-count 256 \
  --input-format bin
```

Custom embedding count:

```bash
python3 vector_db_v3/scripts/pipeline_test.py \
  --build-dir vector_db_v3/build \
  --embedding-count 10000 \
  --input-format bin
```

Smoke mode run (Step 3 behavior):

```bash
python3 vector_db_v3/scripts/pipeline_test.py \
  --build-dir vector_db_v3/build \
  --data-dir /tmp/v3_smoke \
  --embedding-count 128 \
  --runner-smoke
```

Full pipeline run (Step 4/5 behavior):

```bash
python3 vector_db_v3/scripts/pipeline_test.py \
  --build-dir vector_db_v3/build \
  --data-dir /tmp/v3_full \
  --embedding-count 512 \
  --run-full-pipeline \
  --input-format bin
```

Full pipeline with optional stages:

```bash
python3 vector_db_v3/scripts/pipeline_test.py \
  --build-dir vector_db_v3/build \
  --data-dir /tmp/v3_full_opt \
  --embedding-count 512 \
  --run-full-pipeline \
  --with-search-sanity \
  --with-cluster-stats \
  --input-format bin
```

## Expected Output

Stage lifecycle examples:

```text
[START] init
[OK] init | latency_ms=123.456
[FAIL] build-mid-layer-clusters | latency_ms=456.789 | exit=1
```

Cluster summary block (full pipeline):

```text
=== Cluster Summary ===
total_embeddings_inserted: 512
top_cluster_count: 42
mid_cluster_count: 317
lower_branches_continue: 109
lower_branches_stop: 208
final_cluster_count: 208
total_pipeline_latency_ms: 8432.115
```

Results JSON excerpt (`--results-out` file):

```json
{
  "status": "pass",
  "failure_detail": null,
  "timestamp_utc": "2026-03-17T22:10:07.972774+00:00",
  "args": {
    "build_dir": "...",
    "embedding_count": 512,
    "input_format": "bin",
    "run_full_pipeline": true
  },
  "seed_mode": "entropy",
  "stage_results": {
    "runner_smoke": [],
    "full_pipeline": [
      {
        "stage": "init",
        "exit_code": 0,
        "latency_ms": 12.381,
        "command": "..."
      }
    ]
  },
  "cluster_summary": {
    "total_embeddings_inserted": 512,
    "top_cluster_count": 42,
    "mid_cluster_count": 317,
    "lower_branches_continue": 109,
    "lower_branches_stop": 208,
    "final_cluster_count": 208,
    "total_pipeline_latency_ms": 8432.115
  },
  "counts_summary": {
    "inserted": 512,
    "top": 42,
    "mid": 317,
    "lower_continue": 109,
    "lower_stop": 208,
    "final": 208
  },
  "artifacts": {
    "data_dir": "...",
    "dataset_path": "...",
    "results_out": "..."
  },
  "exit_code": 0
}
```

Notes:

- exact latency/count values vary by dataset and machine
- results JSON is written on both success and failure paths

## Exit Codes

- `0`: success
- `1`: runtime/stage failure
- `2`: usage/config validation error

## Troubleshooting

- `error: missing vectordb_v3_cli binary...`
  - verify `--build-dir` points to built CLI output
- `error: --with-search-sanity/--with-cluster-stats require --run-full-pipeline`
  - add `--run-full-pipeline`
- `error: --runner-smoke cannot be combined with --run-full-pipeline`
  - use one mode per run
- summary parse errors after full pipeline
  - check required manifests exist under `<data_dir>/clusters/current/...`

## Notes on Data Paths and Artifacts

- generated dataset path:
  - `bin` -> `<data_dir>/bulk.bin`
  - `jsonl` -> `<data_dir>/bulk.jsonl`
- default results JSON path:
  - `<data_dir>/pipeline_test_results.json`
- full pipeline manifests used for Step 5 summary:
  - `<data_dir>/clusters/current/cluster_manifest.bin`
  - `<data_dir>/clusters/current/mid_layer_clustering/MID_LAYER_CLUSTERING.bin`
  - `<data_dir>/clusters/current/lower_layer_clustering/LOWER_LAYER_CLUSTERING.bin`
  - `<data_dir>/clusters/current/final_layer_clustering/FINAL_LAYER_CLUSTERS.bin`
