# Location: `ProjectX/contract/worker/`

Scope: Immediate contents only (non-recursive).

This folder contains worker-side contract implementation modules:

- `__init__.py`: package-level description of worker contract components.
- `errors.py`: re-exports worker-facing contract exception types.
- `stream_ops.py`: stream/group operations (`ensure_worker_group`, `readgroup`, `autoclaim`).
- `dlq.py`: DLQ append helper (`push_dlq`).
- `contract_client.py`: orchestrator class (`ContractClient`) that coordinates read/reclaim/lifecycle/finalization behavior.
- `runtime.py`: worker runtime wrapper that owns Redis client lifecycle and produces a ready `ContractClient`.
- `scripts/`: folder containing worker-contract Lua script and folder docs.
- `CONTEXT.md`: folder-level context and module inventory for worker contract layer.
- `CONTEXTCHECK.md`: most recent folder context audit report.
- `__pycache__/`: Python bytecode cache artifacts for this subpackage.

