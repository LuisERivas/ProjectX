# worker.md — Redis Streams Worker Notes (Echo Worker)

Contract reference: `contract.md` (authoritative). If conflict, contract wins.

## Deployed Components (systemd)
- `redis-echo-worker.service` (to be created/used)
  - Async worker consuming `jobs:stream` via consumer group.

## Python Redis Client Version (Important)
- Worker should match gateway: prefer `redis==7.2.1` or `redis>=7.2.1,<7.3`.

## Environment variables (worker)
- `REDIS_URL` (default `redis://127.0.0.1:6379/0`)
- `QUEUE_STREAM_KEY` (default `jobs:stream`)
- `WORKER_GROUP` (default `workers`)
- `CONSUMER` (consumer name)
- `BLOCK_MS` (XREADGROUP block time)
- `COUNT` (messages per read)
- `MAX_INFLIGHT` (concurrency cap)
- `DEFAULT_TTL_S` (fallback)

## Echo behavior (current placeholder logic)
- Compute `result_text = f"echo(task={task}): {payload_obj}"`
- Emit optional intermediate event:
  - `type="message"`, `step="worker.echo"`, `data={"text": result_text}`
- Finalize:
  - job hash: `status="done"`, `result={"text":..., "ms":...}`
  - event: `type="done"`, `step="worker.done"`, `data={"ms":...}`

## Critical alignment constraints
- Job state is HASH, not JSON string key.
- Event schema must be `type/ts/step/data` with `data` as JSON string.
- Terminal event must be `done` (or `error`/`canceled`) to stop SSE.

## Planned Next Steps (Architecture Direction)
1. Keep Redis Streams + consumer groups as the backbone.
2. Replace echo logic with a single local model call (same contract).
3. Add orchestration + tools + RAG behind the worker model call.
4. Before model execution, add reliability hardening:
   - pending recovery (XAUTOCLAIM/XCLAIM)
   - idempotency guard (avoid duplicate execution)
   - DLQ stream for poison messages
   - per-job timeout budgeting
   - strict concurrency caps and systemd resource limits
