# Card 5 Rollout / Rollback

## Scope

Card 5 introduces additive async ingest for bulk commands with streamed binary decoding:

- `bulk-insert` and `bulk-insert-bin` remain contract-compatible.
- `bulk-insert-bin` no longer requires full-file materialization before commit.
- Async mode is toggleable and includes CPU-safe fallback.

Runtime toggles:

- `VECTOR_DB_V3_INGEST_ASYNC_MODE=0|1`
- `VECTOR_DB_V3_INGEST_PINNED=0|1`
- `VECTOR_DB_V3_INGEST_FAIL_AFTER_BATCHES=<N>` (validation fault-injection only)

## Rollout Checklist

1. Run `python vector_db_v3/scripts/run_card5_validation.py`.
2. Confirm targeted tests pass:
   - CLI contract and terminal event contract tests
   - durability tests (`wal`, `checkpoint`, `replay_crash`, `corruption`)
   - `vectordb_v3_card5_ingest_async_tests`
3. Confirm command payload/exit semantics remain unchanged.
4. Confirm replay behavior after forced ingest failure matches committed-prefix expectations.
5. Confirm perf checklist passes (or skip perf where explicitly allowed).

## Rollback Strategy

Primary rollback is non-destructive and policy-based:

- Set `VECTOR_DB_V3_INGEST_ASYNC_MODE=0` to force sync ingest path.
- Optionally set `VECTOR_DB_V3_INGEST_PINNED=0`.

This preserves streamed BIN decode improvements while disabling async overlap if needed.

## Emergency Rollback Criteria

Rollback immediately if any of:

- G1/G3/G5/G6/G7 regressions tied to ingest.
- WAL/replay inconsistency under ingest failure scenarios.
- CLI contract drift (payload keys/exit codes/stderr format).
- Sustained ingest performance regression beyond accepted threshold.
