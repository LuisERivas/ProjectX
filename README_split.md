# Split index

This bundle is a split of `agent.md` into smaller, role-scoped files.

- `contract.md` — **Source of truth** for Redis keys, schemas, state transitions, and SSE termination.
- `gateway.md` — Gateway-specific notes (backpressure, deployment assumptions, client version).
- `worker.md` — Worker-specific notes (env vars, echo behavior, planned hardening).
- `runbook.md` — End-to-end verification checklist + ops notes.

Rule: if any file conflicts with `contract.md`, **`contract.md` wins**.
