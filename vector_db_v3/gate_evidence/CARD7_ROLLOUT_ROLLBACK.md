# Card 7 Rollout / Rollback

## Scope

Card 7 introduces a configurable WAL commit policy for ingest-heavy paths:

- `strict_per_record`
- `batch_group_commit`
- `auto` (compatibility-safe default)

The change is additive and does not alter existing CLI payload schemas.

## Runtime Policy

- `VECTOR_DB_V3_WAL_COMMIT_POLICY=auto|strict_per_record|batch_group_commit`
- `auto` preserves current compatibility behavior:
  - single-row operations remain strict,
  - batch ingest paths use grouped commit.

## Rollout Checklist

1. Run `python vector_db_v3/scripts/run_card7_validation.py`.
2. Confirm durability matrix passes for both strict and grouped modes.
3. Confirm CLI/terminal contract tests remain green.
4. Confirm Card 6 composite command remains green under both modes.
5. Confirm evidence artifacts are present and machine-readable.

## Rollback Strategy

Immediate fallback is policy-only and non-destructive:

- Force strict mode:
  - `VECTOR_DB_V3_WAL_COMMIT_POLICY=strict_per_record`

This restores strict commit boundaries while retaining all Card 7 code paths.

## Emergency Rollback Triggers

- Any G2 durability regression.
- Any G6 fail-fast regression in WAL failure/corruption paths.
- Replay/checkpoint inconsistency under crash/corruption tests.
- Contract drift in command outputs or exit codes.
