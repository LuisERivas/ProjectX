# `test_vector_db_combined.py` Options

This guide lists the available options for:

- `scripts/test_vector_db_combined.py`

Run from repository root:

```bash
python scripts/test_vector_db_combined.py --help
```

## Basic Usage

Full flow (configure + build + ctest + generate + smoke/profile):

```bash
python scripts/test_vector_db_combined.py
```

Fast rerun (skip build-related stages):

```bash
python scripts/test_vector_db_combined.py --skip-configure --skip-build --skip-ctest --skip-generate
```

Keep generated run data:

```bash
python scripts/test_vector_db_combined.py --keep-data
```

## Build/Test Stage Flags

- `--skip-configure`  
  Skip `cmake -S ... -B ...`.

- `--skip-build`  
  Skip `cmake --build ...`.

- `--skip-ctest`  
  Skip `ctest --test-dir ...`.

## Data + Smoke Flow Flags

- `--skip-generate`  
  Skip synthetic embedding generation. Requires `--payloads` file to already exist.

- `--payloads <path>`  
  Path to payload JSONL used by `bulk-insert`.  
  Default: `synthetic_dataset_10k_fp16/insert_payloads.jsonl` (resolved under `vector_db/` if relative).

- `--data-dir <path>`  
  Data directory used for smoke/profile run.  
  Default: `smoke_data_combined` (resolved under `vector_db/` if relative).

- `--cluster-seed <seed>`  
  Seed passed to `build-initial-clusters`.  
  Default: `9001`.

## Report + Cleanup Flags

- `--json-out <path>`  
  Output path for combined timing report JSON.  
  Default: `smoke_cli_combined_report.json` (resolved under repo root if relative).

- `--keep-data`  
  Do not delete the run `--data-dir` at the end.

- `--no-clean`  
  Do not delete stale run directories before starting.

## Dependency Preflight Flags

- `--strict-deps`  
  Fail immediately on preflight dependency issues.  
  Default behavior.

- `--no-strict-deps`  
  Record dependency issues in output report and continue execution.

## Synthetic Dataset Generation Flags

These are forwarded to `scripts/generate_synthetic_embeddings.py`:

- `--count <int>` (default: `10000`)
- `--dim <int>` (default: `1024`)
- `--seed <int>` (default: `1337`)
- `--clusters <int>` (default: `32`)
- `--noise-std <float>` (default: `0.04`)
- `--no-normalize` (disable normalization)
- `--gpu-generate` (attempt CuPy-backed generation)
- `--dataset-out <path>` (default: `vector_db/synthetic_dataset_10k_fp16`)
- `--run-second-level` (run second-level clustering validation script after combined smoke flow)
- `--second-level-report <path>` (JSON report output path for second-level validation; default: `vector_db/second_level_test_report.json`)

## Example Combinations

Custom data dir + report path:

```bash
python scripts/test_vector_db_combined.py --data-dir smoke_data_custom --json-out reports/combined_run.json
```

Use existing payloads and skip generation:

```bash
python scripts/test_vector_db_combined.py --skip-generate --payloads synthetic_dataset_10k_fp16/insert_payloads.jsonl
```

Run with non-strict preflight and preserve data:

```bash
python scripts/test_vector_db_combined.py --no-strict-deps --keep-data
```

Run combined flow plus second-level clustering validation:

```bash
python scripts/test_vector_db_combined.py --run-second-level --second-level-report vector_db/second_level_test_report.json
```

## Notes

- On Windows, the script uses `vectordb_cli.exe`; on Unix-like systems it uses `vectordb_cli`.
- Relative `--payloads` and `--data-dir` paths are resolved from `vector_db/`.
- Relative `--json-out` and `--dataset-out` paths are resolved from repo root.

## Related script

For a single-script full rebuild + layer-1 + layer-2 clustering pipeline run with ETA and final cluster summaries, use:

```bash
python scripts/pipeline_test.py
```
