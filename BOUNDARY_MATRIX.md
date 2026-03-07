# Responsibility Matrix

## gateway

- Owns HTTP request/response handling, validation, and transport-level policies.
- Does not perform direct Redis data-plane mutations.
- Uses contract gateway APIs for queue/job/event state operations.

## worker

- Owns runtime orchestration: concurrency limits, scheduling loops, retries, and shutdown behavior.
- Owns business execution logic for jobs.
- Does not perform direct Redis data-plane mutations.

## contract/shared

- Owns shared keys, serialization policy, schema checks, types, and shared config defaults.

## contract/worker

- Owns worker-side Redis contract semantics: read/reclaim, lifecycle transitions, TTL behavior, DLQ, and finalize/ack atomicity.

## contract/gateway

- Owns gateway-side Redis contract semantics: queue/group prep, enqueue workflow, job hash persistence, event stream persistence, backlog metrics, and event stream reads.

## redis

- Serves as storage/stream execution backend only.
- Contains no business routing logic.

