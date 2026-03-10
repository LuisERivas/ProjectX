# ProjectX

ProjectX is a Redis-backed asynchronous job system designed to run on a Jetson host.
It provides an HTTP entrypoint for job submission, a worker process for execution, and
a contract layer that centralizes Redis data-plane behavior.

## What it does

- Accepts jobs over HTTP.
- Stores job state and event history in Redis.
- Processes queued work asynchronously with a worker.
- Streams live job events to clients over SSE.
- Enforces strict component boundaries between gateway, worker, and contract code.

## Core components

- `gateway/`
  - FastAPI HTTP adapter (`POST /v1/jobs`, `POST /v1/jobs/{job_id}/cancel`, `GET /v1/jobs/{job_id}`, `GET /v1/jobs/{job_id}/events`, `GET /health`).
  - Handles request validation, response shaping, SSE formatting, and backpressure policy.
  - Delegates Redis data-plane operations to `contract/gateway/service.py`.

- `worker/`
  - Async runtime loop and concurrency orchestration (`worker_main.py`).
  - Reads and reclaims jobs, runs business logic (currently echo placeholder), handles graceful shutdown.
  - Delegates Redis lifecycle/event/state transitions to `contract/worker/*`.
  - Includes dedicated communications worker (`communications_worker_main.py`) for `sync_communications` and `print_communications` actions.

- `contract/`
  - Canonical Redis contract implementation and shared primitives.
  - `contract/shared/`: config, keys, schema checks, serde, types, error types.
  - `contract/gateway/`: gateway-side Redis operations (`GatewayContract`).
  - `contract/worker/`: worker-side Redis operations (`ContractClient`, stream ops, TTL, DLQ, finalize Lua).

- `Redis`
  - Queue stream: `jobs:stream` (group: `workers`)
  - Communications queue stream: `jobs:communications:stream` (group: `communications-workers`)
  - Job hash: `job:{id}`
  - Per-job event stream: `job:{id}:events`
  - DLQ stream: `jobs:dlq`
  - Optional idempotency mapping: `idempotency:{key}`
  - Communications text key: `communications:text`

## Basic information flow

1. Client calls `POST /v1/jobs` on the gateway.
2. Gateway writes queued job state and queued event, then enqueues message in `jobs:stream`.
3. Worker consumes from `jobs:stream` and marks job `running`.
4. Worker emits intermediate `message` events as needed.
5. Worker finalizes processing to `done` or `error`, while cancellation is a gateway-contract terminal transition (`canceled`) that worker paths honor cooperatively.
6. Client reads final state with `GET /v1/jobs/{job_id}` or streams events via SSE.

Communications flow (dedicated worker):
1. Client submits `task=tool` with payload action `sync_communications` or `print_communications`.
2. Gateway routes those jobs to `jobs:communications:stream`.
3. `redis-communications-worker.service` consumes that stream/group and writes/reads `communications:text`.
4. Client reads printed text from final `result.text`.

## Separation of concerns

- Gateway owns HTTP transport concerns, not direct Redis data-plane logic.
- Worker owns orchestration and execution logic, not ad-hoc Redis mutations.
- Contract packages own Redis schema and state-transition semantics.
- Redis remains the backend store/stream engine with no business routing logic.

Reference: `BOUNDARY_MATRIX.md`

## Current behavior notes

- Worker business logic is currently an echo placeholder in `worker/worker_main.py`.
- Idempotency is supported via `Idempotency-Key` header.
- Backpressure is enforced in the gateway using stream backlog signals.

## Documentation map

- Build tooling prerequisites (Jetson/Linux):
  - Update package index: `sudo apt update`
  - Install GNU C++ compiler and base toolchain: `sudo apt install -y build-essential`
  - Install CMake: `sudo apt install -y cmake`
  - Optional faster CMake generator: `sudo apt install -y ninja-build`

- Jetson setup and deployment: `SETUP.md`
- Functional and boundary validation: `TESTING.md`
- Boundary responsibilities: `BOUNDARY_MATRIX.md`
- Communications client helper: `scripts/communications_client.py`
