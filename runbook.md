# runbook.md — Verification + Ops Notes

Contract reference: `contract.md` (authoritative). If conflict, contract wins.

## Deployed Components (systemd)
- `redis.service`
  - Redis OSS 8.4 running locally only (secured; local-only binding).
- `redis-gateway.service`
  - FastAPI gateway (`main.py`) using `redis.asyncio` client.
- `redis-echo-worker.service` (to be created/used)
  - Async worker consuming `jobs:stream` via consumer group.

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

## File/Service Layout Assumptions (Recommended)
- Gateway directory: `/opt/redis-gateway/` (contains venv + `main.py`)
- Worker directory: `/opt/redis-worker/` (contains venv + `echo_worker.py`)
- Services:
  - `redis.service`
  - `redis-gateway.service`
  - `redis-echo-worker.service`
