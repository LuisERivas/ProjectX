# Card 6 Rollout / Rollback

## Scope

Card 6 introduces an additive single-process full pipeline command:

- New command: `run-full-pipeline`.
- Existing commands remain unchanged (`init`, ingest commands, and per-stage build commands).
- Composite execution preserves stage telemetry semantics while reducing multi-process overhead.

## Rollout Checklist

1. Run `python vector_db_v3/scripts/run_card6_validation.py`.
2. Confirm targeted tests pass:
   - CLI contract and terminal event contract tests
   - durability tests (`wal`, `checkpoint`, `replay_crash`, `corruption`)
   - `vectordb_v3_card6_single_process_tests`
3. Confirm parity report is generated and indicates pass:
   - `parity_report.json`
4. Confirm machine-readable validation evidence exists:
   - `summary.json`, `ctest.log`, `contract_tests.log`
   - optional `perf_report.json` when perf is enabled.
5. Confirm no contract drift in command payloads, exit codes, or stderr format.

## Rollback Strategy

Rollback is immediate and non-destructive by switching orchestration path:

- Keep using legacy stage-by-stage orchestration (`pipeline_test.py --orchestration-mode legacy`).
- Continue using existing per-stage CLI commands directly.

No data migration is required because Card 6 is additive orchestration over existing store/stage behavior.

## Emergency Rollback Criteria

Rollback immediately if any of:

- G1/G3/G5/G6/G7 regressions tied to composite command path.
- Terminal lifecycle ordering/completeness violations in composite mode.
- Deterministic fail-fast behavior differences between composite and legacy mode.
- Sustained performance regression beyond agreed no-regression floor.
