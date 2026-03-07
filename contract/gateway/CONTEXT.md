# Location: `ProjectX/contract/gateway/`

Scope: Immediate contents only (non-recursive).

This folder contains gateway-side contract modules:

- `__init__.py`: package marker/docstring for gateway contract APIs.
- `service.py`: `GatewayContract` implementation that owns Redis data-plane behavior used by the HTTP gateway:
  - worker-group setup
  - job hash persistence/read
  - enqueue flow and queued-event write
  - cancel flow and canceled-event write
  - idempotency mapping write/read
  - backlog metrics and event stream read batches
- `scripts/`: Lua scripts used by gateway contract operations:
  - `create_job.lua`: atomic job creation (idempotency check/set + hash/event/queue write).
  - `cancel_job.lua`: atomic cancel operation (terminal state update + canceled event + TTL handling).
- `CONTEXT.md`: folder-level context and module inventory for the gateway contract layer.
- `CONTEXTCHECK.md`: most recent folder context audit report.
- `__pycache__/`: Python bytecode cache artifacts for this subpackage.

