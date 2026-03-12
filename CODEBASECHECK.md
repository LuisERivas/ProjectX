# CODEBASE Check Report

- **Root folder:** `ProjectX/`
- **Last checked:** `2026-03-11 20:31:45 -07:00`
- **Checker:** `codebase-check`

## Summary

- Files scanned: `96`
- Findings: `4`
- High: `1`
- Medium: `2`
- Low: `1`
- Status: `NEEDS_UPDATE`

## Findings (By Severity)

### HIGH
- Communications worker does not implement cooperative cancellation checks before side effects.
  - **Paths:** `worker/communications_worker_main.py`, `worker/worker_main.py`, `README.md`
  - **Why it matters:** canceled jobs can still write `communications:text` during race windows, violating documented cooperative-cancel semantics and causing surprising state changes.
  - **Recommended fix:** add `is_canceled(job_id)` checks in communications worker before mutating Redis and before finalization, mirroring `worker/worker_main.py`.

### MEDIUM
- Ops guidance drift: remote testing helper omits checks required by `TESTING.md`.
  - **Paths:** `TESTING.md`, `scripts/run_testing_md_remote.py`
  - **Why it matters:** operators can pass the helper while skipping required checks (communications-worker status and CUDA toolchain validation), reducing confidence in readiness.
  - **Recommended fix:** align `run_testing_md_remote.py` output with `TESTING.md` prerequisites/manual follow-ups or explicitly scope the helper in docs.

- Vector DB smoke validation logic is duplicated across multiple scripts.
  - **Paths:** `vector_db/tests/smoke_cli.py`, `vector_db/tests/smoke_cli_profile.py`, `scripts/test_vector_db_combined.py`
  - **Why it matters:** duplicated sequences/assertions can drift, creating conflicting pass/fail behavior between smoke paths.
  - **Recommended fix:** extract shared smoke/assert helpers and reuse them across scripts.

### LOW
- Minor setup wording inconsistency around service count.
  - **Paths:** `SETUP.md`
  - **Why it matters:** docs state the setup script "starts both services" while it provisions/starts three services, which can confuse first-time operators.
  - **Recommended fix:** update wording to "starts all three services."

## Validation Gaps
- No dedicated automated test for communications-worker cancel race semantics (side-effect suppression + terminal consistency).
- No automated test validating gateway backpressure interactions across both queue streams.
- No focused integration test for crash/reclaim edge cases in worker `XAUTOCLAIM` recovery loops.
- No automated guard that setup-generated systemd unit expectations remain synchronized with `SETUP.md` and `TESTING.md`.

## Recommended Next Steps
1. Add cooperative-cancel checks to `worker/communications_worker_main.py` and cover with tests.
2. Align test-ops documentation and helper output so gateway readiness checks are consistent.
3. Run `scripts/run_testing_md_remote.py` and the communications client flow after gateway reachability is restored.
