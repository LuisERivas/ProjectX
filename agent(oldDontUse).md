# agent.md — Jetson Orin Nano Redis Streams Job System (Gateway + Worker Contract)

## Goal
Maintain strict consistency across the Redis-only job gateway and Redis Streams worker(s) on a Jetson Orin Nano (no Docker). This document captures the *current contract* between:
terminal → FastAPI gateway → Redis Streams → worker → Redis → SSE client.

Assume Redis + Gateway are stable and running under systemd.

---

## Deployed Components (systemd)
- `redis.service`
  - Redis OSS 8.4 running locally only (secured; local-only binding).
- `redis-gateway.service`
  - FastAPI gateway (`main.py`) using `redis.asyncio` client.
- `redis-echo-worker.service` (to be created/used)
  - Async worker consuming `jobs:stream` via consumer group.

---

## Python Redis Client Version (Important)
- Gateway venv uses **redis-py 7.2.1** (`pip show redis` => 7.2.1).
- Worker should match gateway: prefer `redis==7.2.1` or `redis>=7.2.1,<7.3`.

This is the *Python client library* version, not Redis server version.

---

## Redis Keys & Data Model (Source of Truth)

### Queue Stream (shared)
- `jobs:stream` (Redis Stream)
  - Gateway enqueues jobs here via `XADD`.
  - Workers consume via consumer group `workers`.

**Queue message fields written by gateway (`XADD jobs:stream`):**
- `job_id` (UUID string)
- `task` (one of: `chat|plan|code|tool|rag|embed`)
- `payload` (JSON string of `CreateJobRequest.payload`)

### Job State (per job)
- `job:{job_id}` (Redis HASH)
  - Gateway stores canonical job record here (not JSON string key).
  - TTL enforced on this key via `EXPIRE job:{job_id} ttl_s`.

**Job hash schema (created by gateway):**
- `job_id` (string)
- `task` (string)
- `payload` (JSON string)  ← stored as JSON string in hash
- `status` (string)        ← `queued|running|done|error|canceled`
- `created_ts` (int ms)
- `updated_ts` (int ms)
- `ttl_s` (int seconds)
- `result` (JSON string or empty)  ← JSON string in hash
- `error` (JSON string or empty)   ← JSON string in hash

Gateway serialization rules:
- Hash JSON fields: `payload`, `result`, `error` are stored as JSON strings.
- Other fields are stored as strings.
- Gateway deserializes these JSON fields back on read.

### Events Stream (per job)
- `job:{job_id}:events` (Redis Stream)
  - SSE endpoint tails this stream using `XREAD`.
  - TTL enforced via `EXPIRE job:{job_id}:events ttl_s` (refreshed on each event).

**Event entry fields (written by gateway + worker):**
- `type` (string)
- `ts` (string int ms)
- `step` (string)
- `data` (JSON string)

**SSE termination condition:**
- SSE stream returns/ends when `type` is one of: `done`, `error`, `canceled`.

This means workers MUST emit a terminal event with `type="done"` (or error/canceled) for clients to stop tailing.

---

## Gateway API Contract (FastAPI)

### POST /v1/jobs
Input schema:
- `task`: `chat|plan|code|tool|rag|embed`
- `payload`: JSON object (max ~200KB)
- `ttl_s`: optional (defaults to `JOB_TTL_S`)

Behavior:
1. Enforces backpressure via backlog (pending + lag), returns 429 if over limit.
2. Creates job hash `job:{id}` with `status="queued"` and TTL.
3. Emits initial event to `job:{id}:events`:  
   - `type="queued"`, `step="gateway.enqueue"`, `data="{}"`
4. Enqueues into `jobs:stream` with fields: `job_id`, `task`, `payload` (payload JSON string).

Idempotency:
- Optional `Idempotency-Key` header maps to `idempotency:{key}` stored with TTL.

### GET /v1/jobs/{id}
- Reads job hash and returns deserialized job object.

### GET /v1/jobs/{id}/events (SSE)
- Validates job exists.
- Emits:
  - `event: hello` at start
  - heartbeat comments periodically
- Tails `job:{id}:events` with `XREAD` from `last_id`.
- Parses event fields and yields SSE:
  - `event: {type}`
  - `data: {json payload}`
- Ends stream if `type` in (`done`, `error`, `canceled`).

---

## Worker Responsibilities (Contract Requirements)

### Consumer group usage
- Stream: `jobs:stream`
- Group: `workers`
- Read via: `XREADGROUP ... streams={jobs:stream: ">"}` to receive new messages
- ACK via: `XACK jobs:stream workers {msg_id}` only after:
  1) job hash state updated, and
  2) terminal event emitted (done/error/canceled)

### Required job state transitions
Worker should:
1. Set job hash `status="running"` and `updated_ts=now_ms()`
2. Emit event `type="running"` (non-terminal)
3. Do work
4. Set job hash `status="done"` (or `error`) and write `result` or `error`
5. Emit terminal event `type="done"` (or `error`)
6. ACK the queue message

### TTL handling
- Worker should read `ttl_s` from job hash field `ttl_s` and use it to `EXPIRE`:
  - `job:{id}`
  - `job:{id}:events`
- TTL should be refreshed on updates/events.

### Payload parsing
Queue message field `payload` is a JSON string. Worker should parse it to an object for processing.
If parsing fails, store `_raw`.

---

## Worker Implementation Notes (Echo Worker - Minimal)

### Environment variables (worker)
- `REDIS_URL` (default `redis://127.0.0.1:6379/0`)
- `QUEUE_STREAM_KEY` (default `jobs:stream`)
- `WORKER_GROUP` (default `workers`)
- `CONSUMER` (consumer name)
- `BLOCK_MS` (XREADGROUP block time)
- `COUNT` (messages per read)
- `MAX_INFLIGHT` (concurrency cap)
- `DEFAULT_TTL_S` (fallback)

### Echo behavior (current placeholder logic)
- Compute `result_text = f"echo(task={task}): {payload_obj}"`
- Emit optional intermediate event:
  - `type="message"`, `step="worker.echo"`, `data={"text": result_text}`
- Finalize:
  - job hash: `status="done"`, `result={"text":..., "ms":...}`
  - event: `type="done"`, `step="worker.done"`, `data={"ms":...}`

### Critical alignment constraints
- Job state is HASH, not JSON string key.
- Event schema must be `type/ts/step/data` with `data` as JSON string.
- Terminal event must be `done` (or `error`/`canceled`) to stop SSE.

---

## Backpressure Model (Gateway)
- Backpressure uses backlog = XPENDING count + optional XINFO GROUPS lag.
- If backlog ≥ `BACKPRESSURE_MAX_BACKLOG` (default 200), gateway returns 429.
- Workers must ACK reliably or backlog grows and gateway rejects new jobs.

---

## Verification Checklist (End-to-End)

### Create job
- POST `/v1/jobs` with JSON body:
  - `{"task":"chat","payload":{"text":"hello"}}`
- Response: `{"job_id": "<uuid>"}`

### SSE tail
- GET `/v1/jobs/{job_id}/events`
- Expected ordered types:
  - `queued` (gateway)
  - `running` (worker)
  - `message` (worker, optional)
  - `done` (worker) → SSE ends

### Job read
- GET `/v1/jobs/{job_id}` returns:
  - `status="done"`
  - `result.text` includes echo output

### Redis ground truth
- `HGETALL job:{job_id}`
- `XRANGE job:{job_id}:events - +`
- `XPENDING jobs:stream workers` should return 0 after completion

---

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

---

## File/Service Layout Assumptions (Recommended)
- Gateway directory: `/opt/redis-gateway/` (contains venv + `main.py`)
- Worker directory: `/opt/redis-worker/` (contains venv + `echo_worker.py`)
- Services:
  - `redis.service`
  - `redis-gateway.service`
  - `redis-echo-worker.service`

End of agent.md.