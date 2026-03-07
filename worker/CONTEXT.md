# Location: `ProjectX/worker/`

Scope: Immediate contents only (non-recursive).

This folder currently contains:

- `worker_main.py`: async worker runtime entrypoint. It handles:
  - Runtime orchestration, signal handling, and concurrency control.
  - Redis runtime startup via `WorkerRuntime`, which provides a ready `ContractClient`.
  - Main consumer loop (`readgroup`) with inflight semaphore control.
  - Pending-message recovery loop (`autoclaim`).
  - Job processing flow (`process_one`) with placeholder echo logic.
  - Lifecycle transitions/events via contract client methods (`mark_running`, `emit_message`, `finalize_done`, `finalize_error`).
  - Graceful shutdown via signal handlers and task cancellation.

- `CONTEXT.md`: folder-level inventory and behavior summary for `worker/`.
- `CONTEXTCHECK.md`: most recent folder context audit report.
- `RECURSIVECONTEXTCHECK.md`: recursive context audit report for `worker/`.
- `__pycache__/`
  - `worker_main.cpython-312.pyc`: Python bytecode cache artifact for `worker_main.py`.

Boundary definition reference: `ProjectX/BOUNDARY_MATRIX.md`

