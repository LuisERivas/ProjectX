# Card 8 Rollout and Rollback: Post-Ingest Checkpoint Shortcut

## Scope

- Feature: post-ingest checkpoint shortcut in `run-full-pipeline`.
- Toggle: `VECTOR_DB_V3_POST_INGEST_CHECKPOINT`.
- Target behavior: reduce reopen/replay burden after ingest-heavy pipeline workflows while preserving durability and fail-fast semantics.

## Rollout Phases

### Phase 1 (MVP)

- Enable shortcut in composite path (`run-full-pipeline`) with env opt-out.
- Require passing:
  - `run_card8_validation.py`
  - durability suites (`wal`, `checkpoint`, `replay_crash`, `corruption`)
  - CLI/terminal event contract tests.

### Phase 2 (Script Orchestration Follow-Up)

- Use legacy script orchestration perf probes with optional `--post-ingest-checkpoint` to compare stage startup latency (`top`, `mid`) and total pipeline latency.

### Phase 3 (Hardening)

- Validate stability on target profile runs and ensure no contract drift in gate evidence.

## Acceptance Thresholds

- Durability/contract matrix: all required suites pass in both shortcut modes (`0`, `1`).
- Performance: no regression beyond `-5.0%` median floor for candidate mode, with one retry allowed for variance.
- Fail-fast: checkpoint failure in the shortcut path must return runtime failure (non-zero).

## Rollback Triggers

- Any G2 durability regression.
- Any CLI/terminal contract drift in required output semantics.
- Repeatable performance regression below floor after retry.

## Safe Fallback

- Immediate operational rollback without code revert:
  - `VECTOR_DB_V3_POST_INGEST_CHECKPOINT=0`
- This preserves pre-Card-8 operational behavior for `run-full-pipeline` checkpoint timing.
