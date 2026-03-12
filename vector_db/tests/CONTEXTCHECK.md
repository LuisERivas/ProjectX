# CONTEXTCHECK

- **Folder:** `vector_db/tests`
- **Last run:** `2026-03-10`
- **Checker:** `context-folder-audit-batch`
- **Scope:** `non-recursive immediate files only`
- **CONTEXT.md dependency:** `disabled`

## Findings
- Reviewed `4` immediate files in this folder.
- No discrepancies detected under immediate-file audit rules.

## Status
- **Result:** `PASS`
- **Issue count:** `0`

## File Summaries
- `benchmark_phase3.py`:
  - Runs a compact benchmark flow against `vectordb_cli` to measure cluster-build throughput and elapsed time.
  - Generates synthetic sparse vectors, builds temporary payloads, executes insert/build commands, and reports JSON metrics.
  - Validates binary existence and runtime command success with explicit error propagation from subprocess output.
  - Prints backend telemetry (`used_cuda`, tensor-core flag, scoring timing) to support quick performance checks.
  - Cleans benchmark data directory at the end to keep test state deterministic across runs.
  - related files: `smoke_cli.py`, `smoke_cli_profile.py`, `../cli/main.cpp`

- `smoke_cli.py`:
  - Executes the integration smoke sequence for the CLI: init, bulk insert, CRUD, WAL checks, checkpoint, clustering, and restart checks.
  - Asserts expected schema and invariants for `cluster-stats`/`cluster-health`, including INT8 elbow telemetry keys.
  - Uses step-indexed logging and helper guards (`require_keys`) to surface stale-binary problems quickly.
  - Verifies persistence behavior by reopening state and checking manifest/version continuity.
  - Serves as the primary confidence test used by higher-level phase scripts.
  - related files: `smoke_cli_profile.py`, `benchmark_phase3.py`, `../cli/main.cpp`, `../src/vector_store.cpp`

- `smoke_cli_profile.py`:
  - Profiles smoke workflow timing per command and writes a ranked JSON timing report for bottleneck analysis.
  - Reuses smoke-style validation logic while capturing elapsed seconds for each subprocess call.
  - Supports runtime options for data directory, payload file, seed, report output, and `--keep-data`.
  - Validates cluster telemetry completeness and restart behavior before finalizing the profile report.
  - Produces both console ranking and structured machine-readable output for regression tracking.
  - related files: `smoke_cli.py`, `benchmark_phase3.py`, `../cli/main.cpp`

- `test_phase1.cpp`:
  - C++ end-to-end test executable covering WAL append/replay, checkpoint truncation, malformed-tail tolerance, and clustering build/reopen behavior.
  - Constructs deterministic vectors and assertions to verify CRUD semantics and persisted state invariants.
  - Exercises initial clustering artifacts and validates cluster stats consistency after reopening the store.
  - Checks bounded elbow trace behavior and existence of generated report/centroid/assignment artifacts.
  - Functions as the CTest-backed compiled regression test for vector DB phase functionality.
  - related files: `../src/vector_store.cpp`, `../include/vector_db/vector_store.hpp`, `../src/clustering/elbow_search.cpp`
