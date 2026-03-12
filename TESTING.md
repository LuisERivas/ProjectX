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

## 12) Vector DB phase test and profiling

These commands validate the Vector DB Phase 3 flow (CMake build, CTest, dataset generation, and CLI smoke checks).

### 12.1 Prerequisites

- Build toolchain installed and available in PATH (`g++`, `cmake`, `ctest`, `nvcc`).
- CUDA-capable runtime is required for `build-initial-clusters`; clustering now fails fast with an explicit error instead of CPU fallback when CUDA/tensor-core execution is unavailable.
- Python available for helper scripts.
- Run from project root unless noted otherwise.

Verify tools:

```bash
g++ --version
cmake --version
ctest --version
nvcc --version
```

### 12.2 Standard Phase test run

From project root:

```bash
python scripts/test_vector_db_phase1.py
```

What it runs:
- CMake configure/build
- CTest for `vectordb_tests`
- Synthetic dataset generation
- CLI smoke integration flow

### 12.3 CLI smoke profiling run (separate script)

From `vector_db/`:

```bash
python tests/smoke_cli_profile.py
```

Optional:

```bash
python tests/smoke_cli_profile.py --keep-data --json-out smoke_cli_profile_report.json
```

Outputs:
- Per-command elapsed times printed in terminal
- Ranked slowest steps
- JSON report with timing breakdown
- Cluster stats now include elbow/stability telemetry fields (`elbow_k_evaluated_count`, `elbow_stage_a_candidates`, `elbow_stage_b_candidates`, `stability_runs_executed`, and per-stage ms timings) for latency validation.

### 12.4 Combined orchestration + smoke/profile script

From project root:

```bash
python scripts/test_vector_db_combined.py
```

For all available options and examples, see:
- `scripts/TEST_VECTOR_DB_COMBINED_OPTIONS.md`

Optional second-level validation in combined flow:

```bash
python scripts/test_vector_db_combined.py --run-second-level
```

### 12.5 Second-level clustering validation script

From project root:

```bash
python scripts/test_vector_db_second_level.py
```

Optional:

```bash
python scripts/test_vector_db_second_level.py --keep-data --json-out vector_db/second_level_test_report.json
```

Outputs:
- Terminal per-centroid processing summary.
- `SECOND_LEVEL_CLUSTERING.json` under:
  - `vector_db/<data_dir>/clusters/initial/v<version>/second_level_clustering/`
- JSON report (default):
  - `vector_db/second_level_test_report.json`
- Per-centroid telemetry includes CUDA/tensor-core usage and INT8/FP16 scoring fields.

### 12.6 Do you need to delete `vector_db/build` each run?

No. For normal testing, reuse `vector_db/build`.

Use a clean rebuild only when:
- toolchain changed (compiler/CUDA/CMake version),
- CMake options or build config changed,
- build artifacts appear stale/corrupted,
- unexplained configure/build failures occur.

Preferred clean order:

```bash
cmake --build vector_db/build --target clean
```

If issues persist, remove the full build directory and reconfigure:

```bash
rm -rf vector_db/build
cmake -S vector_db -B vector_db/build
cmake --build vector_db/build --config Release
```

## 13) Suggested regression checklist

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
