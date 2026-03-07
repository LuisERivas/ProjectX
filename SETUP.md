# SETUP (Jetson)

This guide covers how to set up ProjectX on a Jetson running Ubuntu.

## 1) Prerequisites

- Jetson host with Ubuntu and network access.
- User with `sudo` access.
- Git installed.
- Python 3 with `venv`.

## 2) Clone repository

```bash
cd /home/jetson
git clone <your-repo-url> ProjectX
cd ProjectX
```

## 3) Install system dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip redis curl
```

Enable/start Redis:

```bash
sudo systemctl enable redis
sudo systemctl start redis
sudo systemctl status redis
```

## 4) Create virtual environment and Python dependencies

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn redis==7.2.1 pydantic pytest
```

Optional performance dependency:

```bash
pip install orjson
```

## 5) Configuration

Defaults are already present in code and are suitable for local Jetson deployment.

### Shared variables

- `REDIS_URL` (default `redis://127.0.0.1:6379/0`)
- `QUEUE_STREAM_KEY` (default `jobs:stream`)
- `WORKER_GROUP` (default `workers`)

### Gateway variables

- `JOB_TTL_S` (default `3600`)
- `BACKPRESSURE_MAX_BACKLOG` (default `200`, set `0` to disable)
- `SSE_BLOCK_MS` (default `15000`)
- `SSE_HEARTBEAT_S` (default `15`)

### Worker variables

- `CONSUMER` (default `echo-<pid>`)
- `BLOCK_MS` (default `5000`)
- `COUNT` (default `10`)
- `MAX_INFLIGHT` (default `4`)
- `DEFAULT_TTL_S` (default `3600`)
- `CLAIM_IDLE_MS` (default `60000`)
- `CLAIM_COUNT` (default `25`)
- `CLAIM_EVERY_S` (default `2.0`)
- `JOB_TIMEOUT_S` (default `0`, disabled)

## 6) Run in development mode

Use two terminals with `.venv` activated (run from project root).

Terminal A (gateway):

```bash
cd /home/jetson/ProjectX
python -m uvicorn gateway.main:app --host 0.0.0.0 --port 8000
```

Terminal B (worker):

```bash
cd /home/jetson/ProjectX
python -m worker.worker_main
```

Quick check:

```bash
curl -sS http://127.0.0.1:8000/health
```

## 7) Run as systemd services (recommended)

You can either provision manually or use the helper script.

Note: the current `TESTING.md` workflow assumes systemd-managed services.

### Option A: automated provisioning

From project root:

```bash
python3 jetson_setup.py
```

This script:
- Installs and enables Redis.
- Creates `.venv` and installs Python dependencies.
- Writes and installs:
  - `/etc/systemd/system/redis-gateway.service`
  - `/etc/systemd/system/redis-echo-worker.service`
- Reloads systemd and starts both services.

### Option B: manual service management

```bash
sudo systemctl daemon-reload
sudo systemctl enable redis-gateway.service redis-echo-worker.service
sudo systemctl start redis-gateway.service redis-echo-worker.service
sudo systemctl status redis-gateway.service
sudo systemctl status redis-echo-worker.service
```

## 8) Operational checks

- Redis is healthy: `sudo systemctl status redis`
- Gateway is healthy: `curl -sS http://127.0.0.1:8000/health`
- Worker is running: `sudo systemctl status redis-echo-worker.service`
- Cancel API is reachable (replace with a real job id): `curl -sS -X POST http://127.0.0.1:8000/v1/jobs/<job_id>/cancel -H "Content-Type: application/json" -d "{\"reason\":{\"by\":\"ops\"}}"`

## 9) Common issues

- `ModuleNotFoundError` for `fastapi` or `redis`
  - Activate `.venv` and reinstall dependencies.
- Jobs remain queued
  - Worker not running, wrong env vars, or stream/group mismatch.
- Frequent `429` on job submit
  - Backpressure is active; inspect worker health and queue backlog.
