# CODEBASE Check Report

- **Root folder:** `ProjectX/`
- **Last checked:** `2026-03-10 15:59:01 -07:00`
- **Checker:** `codebase-check`

## Summary

- Files scanned: `93`
- Findings: `5`
- High: `2`
- Medium: `2`
- Low: `1`
- Status: `NEEDS_UPDATE`

## Findings (By Severity)

### HIGH
- Unbounded retry risk on job timeout can keep messages pending and repeatedly reclaimed.
  - **Paths:** `worker/worker_main.py`, `worker/communications_worker_main.py`, `contract/worker/contract_client.py`
  - **Why it matters:** timeout-wrapped execution paths are not consistently terminalized/acked, which can cause replay loops, queue churn, and reliability degradation.
  - **Recommended fix:** explicitly handle timeout at `_process_one_with_timeout` call sites and force deterministic terminal handling (`finalize_error` plus ACK path, with optional DLQ tagging).

- Malformed queue message poison handling lacks guaranteed quarantine+ack path.
  - **Paths:** `contract/worker/contract_client.py`, `contract/shared/schema.py`, `worker/worker_main.py`, `worker/communications_worker_main.py`
  - **Why it matters:** validation failures at envelope conversion can re-enter processing/recovery loops instead of being safely isolated, risking repeated failures.
  - **Recommended fix:** add a poison-message path at dequeue/envelope boundary that captures raw payload, writes to DLQ, and ACKs safely.

### MEDIUM
- Worker runtime resilience differs between main and communications workers due to duplicated orchestration.
  - **Paths:** `worker/worker_main.py`, `worker/communications_worker_main.py`
  - **Why it matters:** one worker loop includes stronger transient exception containment/backoff behavior, while the other can exit on runtime exceptions, creating operational drift.
  - **Recommended fix:** extract shared worker loop/recovery orchestration into a common module and apply one exception/retry contract to both workers.

- Testing documentation claims exceed what the remote runner verifies for communications flow.
  - **Paths:** `TESTING.md`, `scripts/run_testing_md_remote.py`
  - **Why it matters:** mismatch between documented validation scope and actual automation can produce false confidence in communications readiness.
  - **Recommended fix:** either expand `run_testing_md_remote.py` to include communications checks or narrow `TESTING.md` claims to match current runner coverage.

### LOW
- Stale architecture document omits active communications topology.
  - **Paths:** `docs/treeArchitecture`, `README.md`, `worker/communications_worker_main.py`, `contract/shared/config.py`
  - **Why it matters:** onboarding and operations decisions may follow outdated system topology and miss communications components.
  - **Recommended fix:** update `docs/treeArchitecture` to include `redis-communications-worker.service`, `jobs:communications:stream`, and `communications-workers`.

## Validation Gaps
- No automated test proving timeout behavior (`JOB_TIMEOUT_S`) always terminalizes and ACKs.
- No automated poison-message test asserting malformed queue payloads are DLQ'd and ACK'd.
- No test ensuring communications worker loop survives transient Redis/read exceptions.
- No automated backpressure test at gateway backlog threshold.

## Recommended Next Steps
1. Implement deterministic timeout and poison-message handling with guaranteed ACK semantics.
2. Deduplicate worker orchestration into shared runtime loop logic to remove resilience drift.
3. Add targeted integration tests (timeout, poison, communications resilience, backpressure), then rerun `codebase-check`.
