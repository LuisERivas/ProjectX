# Terminal Event Contract

## Purpose

Define machine-parseable terminal telemetry emitted during pipeline execution.

## Transport

- Required format: JSONL on stdout.
- Default behavior: enabled for all pipeline runs (not debug-only).

## Event Types

- `pipeline_start`
- `stage_start`
- `stage_end`
- `stage_fail`
- `stage_skip`
- `pipeline_summary`

Optional:

- `stage_progress`
- `compliance_check`
- `artifact_write`
- `k_selection`

## Required Common Fields

- `event_type`
- `stage_id`
- `stage_name`
- `status`
- `start_ts`
- `end_ts` (required for end/fail/skip)
- `elapsed_ms`
- `pipeline_elapsed_ms`
- `active_pipeline_state`

## Conditional Required Fields

Failure events must include:

- `error_code`
- `error_message`

Hardware non-compliance events must include:

- `non_compliance_stage`

Per-centroid or per-cluster events must include:

- `centroid_id` and/or `job_id`

Reporting baseline fields:

- `stage_started_ts` (required for stage lifecycle events)
- `stage_elapsed_ms` (required for progress/end/fail)
- `previous_run_stage_elapsed_ms` (required when baseline exists, nullable otherwise)
- `previous_run_available` (bool)

Precision alignment reporting fields (required when precision artifacts are validated in stage scope):

- `source_embedding_artifact`
- `compute_precision`
- `alignment_check_status` (`pass|fail`)
- `alignment_mismatch_count`
- `alignment_failure_reason` (required when `alignment_check_status=fail`)

K-selection events must include:

- `k_min`
- `k_max`
- `chosen_k`
- `tested_ks`

## Ordering and Validity Rules

- Events are emitted in execution order.
- Every `stage_start` has exactly one matching terminal event (`stage_end`, `stage_fail`, or `stage_skip`).
- `elapsed_ms` and `pipeline_elapsed_ms` are non-negative.
- `pipeline_elapsed_ms` is monotonic within a run.



