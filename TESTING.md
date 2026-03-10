# TESTING

This guide combines automated smoke checks and manual validation steps for ProjectX.
The remote runner covers a core API subset only; full end-to-end confidence still requires the manual sections below.

## 1) Scope note

For now, testing is standardized on **systemd mode only**.

Dev-mode test flow will be expanded in a future update and can be ignored for now.

## 2) Remote automated runner (no SSH)

If you are testing from another machine, you can run the remote helper script that prompts for Jetson IP and executes non-disruptive core API checks from this document.

From project root:

```bash
python scripts/run_testing_md_remote.py
```

Optional flags:

```bash
python scripts/run_testing_md_remote.py --host 192.168.1.50 --port 8000 --timeout-s 8 --poll-timeout-s 45
```

What it automates:
- Health check (`GET /health`)
- Submit + read terminal state (`POST /v1/jobs`, `GET /v1/jobs/{job_id}`)
- Idempotency check (`Idempotency-Key` reused)
- Cancel endpoint check (`POST /v1/jobs/{job_id}/cancel`)

What it does not automate:
- SSE event progression checks
- Redis ground-truth checks with `redis-cli`
- Recovery behavior checks
- Communications worker flow (`scripts/communications_client.py`)
- Boundary static test (`python -m pytest tests/test_boundary_rules.py`)

What remains manual without SSH:
- `systemctl` status checks on Jetson
- Toolchain checks (`g++`, `cmake`, `nvcc`)

Exit codes:
- `0`: all automated checks passed
- `1`: one or more automated checks failed

## 3) Test prerequisites

- Redis running locally (`redis://127.0.0.1:6379/0` by default).
- Gateway running as `redis-gateway.service`.
- Worker running as `redis-echo-worker.service`.
- Communications worker running as `redis-communications-worker.service`.
- `curl` and `redis-cli` available.
- C++/CUDA build toolchain available on Jetson (`g++`, `cmake`, `nvcc`) for local vector DB build/test steps.

Ensure services are active:

```bash
sudo systemctl status redis
sudo systemctl status redis-gateway.service
sudo systemctl status redis-echo-worker.service
sudo systemctl status redis-communications-worker.service
```

Verify build tools are installed:

```bash
g++ --version
cmake --version
nvcc --version
```

## 4) Health check

```bash
curl -sS http://127.0.0.1:8000/health
```

Expected:
- `"ok": true`
- Additional health fields may be present depending on runtime configuration

## 5) API job flow

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

## 6) Idempotency behavior

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

## 7) Cancel behavior

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

## 8) Redis ground-truth checks

```bash
redis-cli -u redis://127.0.0.1:6379/0 HGETALL "job:${JOB_ID}"
redis-cli -u redis://127.0.0.1:6379/0 XRANGE "job:${JOB_ID}:events" - +
redis-cli -u redis://127.0.0.1:6379/0 XPENDING "jobs:stream" "workers"
```

Expected:
- Job hash exists and reflects terminal status.
- Events stream includes lifecycle transitions.
- `XPENDING` trends to zero after job completion.

## 9) Recovery behavior check (systemd mode only)

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

## 10) Communications worker flow

Prepare root file:

```bash
cat > communications.txt <<'EOF'
hello from communications worker
EOF
```

Run client helper:

```bash
python scripts/communications_client.py --host 127.0.0.1 --port 8000
```

Expected:
- First submitted job (`sync_communications`) reaches `done`.
- Second submitted job (`print_communications`) reaches `done`.
- Script prints text that matches `communications.txt`.
- Redis value exists:

```bash
redis-cli -u redis://127.0.0.1:6379/0 GET communications:text
```

## 11) Boundary rule tests (static guardrails)

From project root, run:

```bash
python -m pytest tests/test_boundary_rules.py
```

Expected:
- Tests pass.
- Confirms no forbidden direct Redis data-plane calls were introduced in:
  - `gateway/main.py`
  - `worker/worker_main.py`

## 12) Suggested regression checklist

Run this minimum suite after code changes (automated + manual):
- Verify systemd services are active
- `GET /health`
- Submit/read one job
- Stream one job over SSE (manual)
- Idempotency test
- Cancel test
- Communications worker sync/print test
- Redis ground-truth checks
- `pytest tests/test_boundary_rules.py`
