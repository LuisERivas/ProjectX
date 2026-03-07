# CODEBASE Check Report

- **Root folder:** `ProjectX/`
- **Last checked:** `2026-03-06 16:00:20 -08:00`
- **Checker:** `codebase-check`

## Summary

- Files scanned: `61`
- Findings: `3`
- High: `2`
- Medium: `0`
- Low: `1`
- Status: `NEEDS_UPDATE`

## Findings (By Severity)

### HIGH
- Timeout path can leave jobs pending and repeatedly reclaimed.
  - **Paths:** `worker/worker_main.py`, `contract/worker/contract_client.py`
  - **Why it matters:** `asyncio.wait_for` timeout in task execution is not finalized/acked in the timeout path, so jobs can loop through reclaim/retry indefinitely.
  - **Recommended fix:** catch timeout explicitly in worker task execution and route to deterministic terminal handling (`finalize_error` or DLQ + ACK).

- Malformed queue messages can stall processing without DLQ/ACK.
  - **Paths:** `contract/worker/contract_client.py`, `contract/shared/schema.py`, `worker/worker_main.py`
  - **Why it matters:** schema validation failures in `_to_envelope` bubble to run loop, which backs off but does not quarantine/ack the bad message.
  - **Recommended fix:** add poison-message handling path (DLQ + ACK) for envelope/schema errors.

### LOW
- DLQ stream name is hardcoded while queue/group are env-configurable.
  - **Paths:** `contract/shared/redis_keys.py`, `contract/shared/config.py`
  - **Why it matters:** environment namespacing can drift for DLQ in multi-env deployments.
  - **Recommended fix:** move DLQ key default into shared config and reference from key helper.

## Validation Gaps
- No automated Redis test for timeout/reclaim + malformed-message poison handling.
- No automated check ensures docs/check reports remain synchronized after rapid code changes.
- Local host Python session still lacks `pytest`; tests are expected to run from project venv.

## Recommended Next Steps
1. Add deterministic timeout handling path (terminalize or DLQ + ACK) in worker execution wrapper.
2. Add poison-message handling for schema/envelope failures (DLQ + ACK).
3. Add integration tests for timeout + poison-message quarantine/ack behavior, then re-run `codebase-check`.
