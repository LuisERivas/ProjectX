# Testing Implementation Plan: `vector_db_v3` Synthetic Pipeline Test

## Purpose

This document is a copy/paste-ready implementation plan you can hand to a plan/implementation agent to build a new pipeline test script for `vector_db_v3` that:

- Lets the user choose how many synthetic embeddings to generate
- Uses truly random embeddings
- Runs the full clustering pipeline (top -> mid -> lower -> final)
- Prints per-step latency in terminal output
- Produces a machine-readable summary file for regression/performance tracking

---

## Scope Lock (Do Not Break Existing Behavior)

- Add new functionality only (new script + optional CMake wiring).
- Do not change contract behavior of existing commands.
- Do not change artifact schemas/contracts.
- Do not alter gate semantics for existing G1..G7 tests.
- Keep existing scripts (`run_gates.py`, `perf_gate.py`, `check_reproducibility.py`) working as-is.

---

## Target Deliverables

1. New script: `vector_db_v3/scripts/pipeline_test.py`
2. New output artifact from script: `pipeline_test_results.json`
3. Clear terminal output with per-stage latency and summary counts

---

## Implementation Plan (Step-by-Step)

## Step 1: Add Script Skeleton and CLI Interface

### Files
- `vector_db_v3/scripts/pipeline_test.py` (new)

### Requirements
- Add CLI args:
  - `--build-dir` (required, path to build dir containing `vectordb_v3_cli`)
  - `--data-dir` (optional, default temp location)
  - `--embedding-count` (required, integer > 0)
  - `--results-out` (optional path for JSON summary)
  - `--keep-data` (optional flag to preserve generated files)

### Validation Rules
- Fail early with clear `error:` messages if:
  - `--embedding-count <= 0`
  - CLI binary not found in `--build-dir`
  - `--batch-size <= 0`

### Acceptance
- Running `python pipeline_test.py --help` prints all arguments and usage.

---

## Step 2: Implement Synthetic Embedding Generation

### Files
- `vector_db_v3/scripts/pipeline_test.py`

### Requirements
- Generate `embedding_count` vectors with dimension 1024.
- JSONL row format:
  - `{"embedding_id": <u64>, "vector": [<1024 floats>]}`
- Random policy:
  - Entropy-based random generation per run
- Write to `<data_dir>/bulk.jsonl`.

### Notes
- Keep generation memory-efficient:
  - Stream rows directly to file, do not hold full dataset in memory.

### Acceptance
- For `--embedding-count N`, JSONL has exactly `N` non-empty rows.

---

## Step 3: Implement Command Runner + Latency Measurement

### Files
- `vector_db_v3/scripts/pipeline_test.py`

### Requirements
- Add helper to run subprocess commands and capture:
  - exit code
  - stdout
  - stderr
  - elapsed milliseconds (using `time.perf_counter`)
- Normalize command failure handling:
  - print concise failure summary
  - stop pipeline on first failure (fail-fast)

### Terminal Print Format (example)
- `[START] init`
- `[OK] init | latency_ms=123.456`
- `[FAIL] build-mid-layer-clusters | latency_ms=456.789 | exit=1`

### Acceptance
- Each command stage logs start/end with latency.

---

## Step 4: Run Full Pipeline Stages

### Files
- `vector_db_v3/scripts/pipeline_test.py`

### Required Stage Order
1. `init`
2. `bulk-insert`
3. `build-top-clusters`
4. `build-mid-layer-clusters`
5. `build-lower-layer-clusters`
6. `build-final-layer-clusters`
7. optional `search` sanity command
8. optional `cluster-stats`

### Acceptance
- Script enforces order and fails cleanly if any stage fails.

---

## Step 5: Parse and Print Cluster Summary

### Files
- `vector_db_v3/scripts/pipeline_test.py`

### Requirements
- Read produced pipeline outputs and print:
  - total embeddings inserted
  - top cluster count
  - mid cluster count
  - lower branch counts (`stop` / `continue`)
  - final cluster count
  - total pipeline latency
- Prefer using existing stage outputs/manifests already emitted under:
  - `<data_dir>/clusters/current/...`

### Acceptance
- End-of-run summary includes both latency and cluster breakdown.

---

## Step 6: Write Machine-Readable Results File

### Files
- `vector_db_v3/scripts/pipeline_test.py`

### Requirements
- Write JSON summary to `--results-out` (or default `<data_dir>/pipeline_test_results.json`).
- Include:
  - run metadata (timestamp, args used, seed mode)
  - per-stage latency + exit status
  - counts summary (top/mid/lower/final/inserted)
  - overall status (`pass`/`fail`)
  - failure detail if failed

### Acceptance
- JSON is always written (even on failure) with status and diagnostics.

---

## Step 7: Documentation Update (Optional but Recommended)

### Files
- `vector_db_v3/scripts/README.md` (if present), or create small usage section in a suitable existing doc

### Requirements
- Add usage examples:
  - deterministic run with seed
  - entropy run without seed
  - custom embedding count
- Include expected output examples.

---

## Regression Safety Checklist

Before closing implementation:

- Existing `scripts/test_cli_contract.py` still passes.
- Existing `scripts/test_terminal_event_contract.py` still passes.
- Existing clustering artifact tests pass:
  - `top_layer_artifacts`
  - `mid_layer_artifacts`
  - `lower_layer_gate_artifacts`
  - `final_layer_artifacts`
  - `final_layer_eligibility_reconciliation`
- Repro script remains unaffected.
- No changes to existing contract files unless explicitly requested.

---

## Suggested Execution Sequence for Agent

1. Implement Steps 1-3 (CLI + generator + timing runner).
2. Implement Step 4 (pipeline execution order).
3. Implement Steps 5-6 (summary parsing + results JSON).
4. Run local validations on smoke-size count.
5. Final verification with existing contract tests (regression safety checklist).

---

