# Location: `ProjectX/gateway/`

Scope: Immediate contents only (non-recursive).

This folder currently contains:

- `main.py`: FastAPI gateway entrypoint and HTTP adapter. It handles:
  - Request validation and response shaping for job endpoints.
  - Delegation of Redis data-plane operations to `GatewayContract` in `contract/gateway/service.py`.
  - Backpressure policy checks and SSE formatting/stream response behavior.

- `CONTEXT.md`: folder-level inventory and behavior summary for `gateway/`.
- `CONTEXTCHECK.md`: most recent folder context audit report.
- `RECURSIVECONTEXTCHECK.md`: recursive context audit report for `gateway/`.
- `__pycache__/`
  - `main.cpython-312.pyc`: Python bytecode cache artifact for `main.py`.

Boundary definition reference: `ProjectX/BOUNDARY_MATRIX.md`

