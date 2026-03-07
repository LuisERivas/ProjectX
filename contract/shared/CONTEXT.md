# Location: `ProjectX/contract/shared/`

Scope: Immediate contents only (non-recursive).

This folder contains shared contract primitives:

- `__init__.py`: summary docstring for the shared primitives package.
- `config.py`: shared environment-backed defaults (`REDIS_URL`, `QUEUE_STREAM_KEY`, `WORKER_GROUP`).
- `errors.py`: base exception types (`ContractError`, `ContractViolation`, `SchemaError`, `OrderingError`).
- `redis_keys.py`: key builder helpers (`job_key`, `events_key`, `dlq_stream_key`).
- `job_hash.py`: shared hash serialization/deserialization policy for job hash mappings.
- `group_ops.py`: canonical worker-group initialization helper (`ensure_worker_group`).
- `schema.py`: queue message validation helper (`validate_queue_message`).
- `serde.py`: JSON serialization and deserialization helpers (`dumps`, `loads`) with optional `orjson`.
- `types.py`: dataclasses used for queue and job envelopes (`QueueMessage`, `JobEnvelope`).
- `CONTEXT.md`: folder-level inventory for shared contract primitives.
- `CONTEXTCHECK.md`: most recent folder context audit report.
- `__pycache__/`: Python bytecode cache artifacts for this subpackage.

