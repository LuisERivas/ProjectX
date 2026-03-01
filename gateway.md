# gateway.md — FastAPI Redis Gateway Notes

Contract reference: `contract.md` (authoritative). If conflict, contract wins.

## Deployed Components (systemd)
- `redis-gateway.service`
  - FastAPI gateway (`main.py`) using `redis.asyncio` client.

## Python Redis Client Version (Important)
- Gateway venv uses **redis-py 7.2.1** (`pip show redis` => 7.2.1).

This is the *Python client library* version, not Redis server version.

## Backpressure Model
- Backpressure uses backlog = XPENDING count + optional XINFO GROUPS lag.
- If backlog ≥ `BACKPRESSURE_MAX_BACKLOG` (default 200), gateway returns 429.
- Workers must ACK reliably or backlog grows and gateway rejects new jobs.

## API Endpoints (Summary)
See `contract.md` for the authoritative behavioral contract:
- POST `/v1/jobs`
- GET `/v1/jobs/{id}`
- GET `/v1/jobs/{id}/events` (SSE)

## File/Service Layout Assumptions (Recommended)
- Gateway directory: `/opt/redis-gateway/` (contains venv + `main.py`)
- Services:
  - `redis.service`
  - `redis-gateway.service`
