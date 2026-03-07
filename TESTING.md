# TESTING

This guide validates that ProjectX works end-to-end and that architectural boundaries remain intact.

## 1) Scope note

For now, testing is standardized on **systemd mode only**.

Dev-mode test flow will be expanded in a future update and can be ignored for now.

## 2) Test prerequisites

- Redis running locally (`redis://127.0.0.1:6379/0` by default).
- Gateway running as `redis-gateway.service`.
- Worker running as `redis-echo-worker.service`.
- `curl` and `redis-cli` available.

Ensure services are active:

```bash
sudo systemctl status redis-server
sudo systemctl status redis-gateway.service
sudo systemctl status redis-echo-worker.service
```

## 3) Health check

```bash
curl -sS http://127.0.0.1:8000/health
```

Expected:
- `"ok": true`
- `queue_stream` and `worker_group` fields present

## 4) End-to-end job flow

### 4.1 Submit a job

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/jobs" \
  -H "Content-Type: application/json" \
  -d "{\"task\":\"chat\",\"payload\":{\"text\":\"hello from test\"}}"
```

Expected:
- JSON response with `job_id`

Save it:

```bash
JOB_ID="<paste-job-id>"
```

### 4.2 Stream events (SSE)

```bash
curl -N "http://127.0.0.1:8000/v1/jobs/${JOB_ID}/events"
```

Expected event progression:
- `hello`
- `queued`
- `running`
- `message` (may be one or more)
- terminal event `done` (or `error`)

### 4.3 Read final job state

```bash
curl -sS "http://127.0.0.1:8000/v1/jobs/${JOB_ID}"
```

Expected:
- `status` is terminal (`done` for normal echo path)
- `result` populated on success

## 5) Idempotency behavior

Run twice with the same idempotency key:

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/jobs" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: demo-123" \
  -d "{\"task\":\"chat\",\"payload\":{\"text\":\"same request\"}}"
```

Expected on second call:
- `"idempotent": true`
- Same `job_id` as the first call

## 6) Cancel behavior

Cancel the submitted job:

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/jobs/${JOB_ID}/cancel" \
  -H "Content-Type: application/json" \
  -d "{\"reason\":{\"by\":\"test\"}}"
```

Expected:
- Response includes `status` (`canceled` or existing terminal status).
- If cancellation wins before completion, SSE terminal event is `canceled`.
- If completion already happened, status remains existing terminal state.

## 7) Redis ground-truth checks

```bash
redis-cli -u redis://127.0.0.1:6379/0 HGETALL "job:${JOB_ID}"
redis-cli -u redis://127.0.0.1:6379/0 XRANGE "job:${JOB_ID}:events" - +
redis-cli -u redis://127.0.0.1:6379/0 XPENDING "jobs:stream" "workers"
```

Expected:
- Job hash exists and reflects terminal status.
- Events stream includes lifecycle transitions.
- `XPENDING` trends to zero after job completion.

## 8) Recovery behavior check (systemd mode only)

1. Stop worker:

```bash
sudo systemctl stop redis-echo-worker.service
```

2. Submit a job (it should remain queued/pending).
3. Start worker again:

```bash
sudo systemctl start redis-echo-worker.service
```

Expected:
- Worker recovers and processes pending work.

## 9) Boundary rule tests (static guardrails)

From project root, run:

```bash
python -m pytest tests/test_boundary_rules.py
```

Expected:
- Tests pass.
- Confirms no forbidden direct Redis data-plane calls were introduced in:
  - `gateway/main.py`
  - `worker/worker_main.py`

## 10) Suggested regression checklist

Run this minimum suite after code changes:
- Verify systemd services are active
- `GET /health`
- Submit/read/stream one job
- Idempotency test
- Cancel test
- Redis ground-truth checks
- `pytest tests/test_boundary_rules.py`
