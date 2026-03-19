# Card 4 Rollout / Rollback

## Scope

Card 4 adds internal embedding shards:

- Canonical: `embeddings_fp32.bin`
- Derived: `embeddings_fp16.bin`
- Derived: `embeddings_int8_sym.bin`

Runtime selection is controlled via:

- `VECTOR_DB_V3_INTERNAL_SHARD_MODE=off|auto|fp16|int8|strict`
- `VECTOR_DB_V3_INTERNAL_SHARD_REPAIR=regenerate|fallback|fail`

## Rollout Checklist

1. Run `python vector_db_v3/scripts/run_card4_validation.py`.
2. Confirm targeted tests pass, including:
   - `vectordb_v3_precision_shard_lifecycle_tests`
   - `vectordb_v3_precision_shard_alignment_failures_tests`
   - `vectordb_v3_terminal_event_contract_tests`
3. Confirm Card 4 telemetry fields are present in terminal events:
   - `source_embedding_artifact`
   - `compute_precision`
   - `alignment_check_status`
   - `alignment_mismatch_count`
   - `precision_fallback_reason`
4. Confirm no regressions in G1/G3/G5/G6/G7.
5. If perf checks are enabled, confirm no major (>3%) top/mid regression.

## Rollback Strategy

Primary rollback is policy-only (no code revert required):

- Set `VECTOR_DB_V3_INTERNAL_SHARD_MODE=off`.
- Optionally set `VECTOR_DB_V3_INTERNAL_SHARD_REPAIR=fallback`.

This reverts stage compute to canonical FP32 shard selection while keeping additive artifacts harmless.

## Emergency Rollback Criteria

Rollback immediately if any of the following occurs:

- Any G1/G3/G5/G6/G7 hard failure caused by shard selection/lifecycle.
- Reproducible alignment failure without deterministic regenerate/fallback behavior.
- Persistent stage failures caused by shard corruption handling in production profile.
- Card 4 perf checks show sustained top/mid degradation beyond accepted threshold.
