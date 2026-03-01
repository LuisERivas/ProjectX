# contract.md — Redis Streams Job System Contract (Source of Truth)

## Goal
Maintain strict consistency across the Redis-only job gateway and Redis Streams worker(s) on a Jetson Orin Nano (no Docker). This document is the **authoritative contract** between:
terminal → FastAPI gateway → Redis Streams → worker → Redis → SSE client.

If anything in other docs conflicts with this file, **this file wins**.

---

## Redis Keys & Data Model

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

**Serialization rules:**
- Hash JSON fields: `payload`, `result`, `error` are stored as JSON strings.
- Other fields are stored as strings.
- Readers must deserialize `payload/result/error` on read.

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
- SSE ends when `type` is one of: `done`, `error`, `canceled`.

Workers **MUST** emit a terminal event with `type="done"` (or `error`/`canceled`) for clients to stop tailing.

---

## Gateway API Contract (Behavioral)

### POST `/v1/jobs`
Input schema:
- `task`: `chat|plan|code|tool|rag|embed`
- `payload`: JSON object (max ~200KB)
- `ttl_s`: optional (defaults to `JOB_TTL_S`)

Behavior:
1. Enforces backpressure (implementation can vary; must be functionally equivalent).
2. Creates job hash `job:{id}` with `status="queued"` and TTL.
3. Emits initial event to `job:{id}:events`:
   - `type="queued"`, `step="gateway.enqueue"`, `data="{}"`
4. Enqueues into `jobs:stream` with fields: `job_id`, `task`, `payload` (payload JSON string).

Idempotency:
- Optional `Idempotency-Key` header maps to `idempotency:{key}` stored with TTL.

### GET `/v1/jobs/{id}`
- Reads job hash and returns deserialized job object.

### GET `/v1/jobs/{id}/events` (SSE)
- Validates job exists.
- Emits:
  - `event: hello` at start
  - heartbeat comments periodically
- Tails `job:{id}:events` with `XREAD` from `last_id`.
- Yields SSE:
  - `event: {type}`
  - `data: {json payload}`
- Ends stream if `type` in (`done`, `error`, `canceled`).

---

## Worker Contract Requirements

### Consumer group usage
- Stream: `jobs:stream`
- Group: `workers`
- Read via: `XREADGROUP ... streams={jobs:stream: ">"}`
- ACK via: `XACK jobs:stream workers {msg_id}` **only after**:
  1) job hash state updated, and
  2) terminal event emitted (`done/error/canceled`)

### Required job state transitions
Worker should:
1. Set job hash `status="running"` and `updated_ts=now_ms()`
2. Emit event `type="running"` (non-terminal)
3. Do work
4. Set job hash `status="done"` (or `error`) and write `result` or `error`
5. Emit terminal event `type="done"` (or `error`)
6. ACK the queue message

### TTL handling
- Worker should read `ttl_s` from job hash field `ttl_s` and `EXPIRE`:
  - `job:{id}`
  - `job:{id}:events`
- TTL should be refreshed on updates/events.

### Payload parsing
- Queue message field `payload` is a JSON string.
- Worker should parse it to an object for processing.
- If parsing fails, store `_raw`.
