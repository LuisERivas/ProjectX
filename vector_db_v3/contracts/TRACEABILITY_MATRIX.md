# Traceability Matrix

## Purpose

Provide requirement-to-contract-to-test mapping for M1 implementation governance.

## Matrix

| Requirement Area | Contract File | Primary Validation Gate |
|---|---|---|
| Exact-only search; no ANN | `M1_SCOPE_CONTRACT.md`, `CLI_CONTRACT.md` | G1, G3 |
| No metadata filter/ranking | `M1_SCOPE_CONTRACT.md`, `CLI_CONTRACT.md` | G1, G3 |
| Single active clustering state | `M1_SCOPE_CONTRACT.md`, `ARTIFACT_CONTRACT.md` | G2, G3 |
| Stage order Top->Mid->Lower->Final | `M1_SCOPE_CONTRACT.md`, `TEST_GATE_CONTRACT.md` | G1, G7 |
| Lower gate semantics | `M1_SCOPE_CONTRACT.md`, `TEST_GATE_CONTRACT.md` | G1 |
| Final eligibility (gate-stop only) | `M1_SCOPE_CONTRACT.md`, `ARTIFACT_CONTRACT.md`, `TEST_GATE_CONTRACT.md` | G1, G3 |
| Final per-cluster artifacts | `ARTIFACT_CONTRACT.md` | G3 |
| Binary-first assignment artifacts and byte layouts | `M1_SCOPE_CONTRACT.md`, `ARTIFACT_CONTRACT.md`, `BINARY_FORMATS.md` | G3 |
| Binary `id_estimate` and end-of-pipeline k-search bounds batch artifact | `M1_SCOPE_CONTRACT.md`, `ARTIFACT_CONTRACT.md`, `TEST_GATE_CONTRACT.md` | G3, G7 |
| End-of-pipeline consolidated post-cluster membership artifact | `ARTIFACT_CONTRACT.md`, `BINARY_FORMATS.md`, `TEST_GATE_CONTRACT.md` | G3 |
| Precision consistency by ID-alignment across FP32/FP16/INT8 artifacts (no quant sidecar metadata requirement) | `ARTIFACT_CONTRACT.md`, `BINARY_FORMATS.md`, `TERMINAL_EVENT_CONTRACT.md`, `TEST_GATE_CONTRACT.md` | G3, G7 |
| Internal precision shard lifecycle (FP32 canonical + FP16/INT8 derived) and deterministic repair/fallback behavior | `ARTIFACT_CONTRACT.md`, `BINARY_FORMATS.md`, `TEST_GATE_CONTRACT.md` | G1, G3, G6 |
| Card 4 validation runner evidence (`run_card4_validation.py`) for targeted correctness + AB checks | `TEST_GATE_CONTRACT.md`, `TRACEABILITY_MATRIX.md` | G1, G3, G4, G5, G6, G7 |
| Async ingest pipeline parity and streamed binary ingest behavior (`bulk-insert-bin` without full materialization) | `CLI_CONTRACT.md`, `M1_SCOPE_CONTRACT.md`, `TEST_GATE_CONTRACT.md` | G1, G3 |
| Async ingest durability/failure-boundary replay behavior | `M1_SCOPE_CONTRACT.md`, `TEST_GATE_CONTRACT.md` | G1, G2, G6 |
| Card 5 validation runner evidence (`run_card5_validation.py`) for ingest build/test/perf suite | `TEST_GATE_CONTRACT.md`, `TRACEABILITY_MATRIX.md` | G1, G3, G4, G5, G6, G7 |
| Single-process full pipeline composite command parity against legacy stage-by-stage orchestration | `CLI_CONTRACT.md`, `TERMINAL_EVENT_CONTRACT.md`, `TEST_GATE_CONTRACT.md` | G1, G3, G7 |
| Card 6 validation runner evidence (`run_card6_validation.py`) for build/test/parity/perf suite | `TEST_GATE_CONTRACT.md`, `TRACEABILITY_MATRIX.md` | G1, G3, G4, G5, G6, G7 |
| WAL commit policy matrix for ingest/batch durability boundaries (`strict_per_record` vs `batch_group_commit`) | `M1_SCOPE_CONTRACT.md`, `CLI_CONTRACT.md`, `TEST_GATE_CONTRACT.md` | G1, G2, G3, G6 |
| Card 7 validation runner evidence (`run_card7_validation.py`) for durability matrix and perf comparison | `TEST_GATE_CONTRACT.md`, `TRACEABILITY_MATRIX.md` | G1, G2, G3, G4, G5, G6, G7 |
| Post-ingest checkpoint shortcut policy for pipeline workflows (`VECTOR_DB_V3_POST_INGEST_CHECKPOINT`) | `CLI_CONTRACT.md`, `TEST_GATE_CONTRACT.md` | G1, G2, G3, G6 |
| Card 8 validation runner evidence (`run_card8_validation.py`) for durability matrix and checkpoint A/B performance comparison | `TEST_GATE_CONTRACT.md`, `TRACEABILITY_MATRIX.md` | G1, G2, G3, G4, G5, G6, G7 |
| Terminal JSONL lifecycle events | `TERMINAL_EVENT_CONTRACT.md` | G7 |
| Active pipeline terminal reporting and previous-run timing baseline | `TERMINAL_EVENT_CONTRACT.md` | G7 |
| CUDA/Tensor/Ampere compliance | `COMPLIANCE_CONTRACT.md` | G5, G6 |
| Tensor Core FP16 distance-path selection truthfulness and fallback evidence (`cuda_tensor_fp16` vs `cuda_fp32`) | `COMPLIANCE_CONTRACT.md`, `TERMINAL_EVENT_CONTRACT.md`, `TEST_GATE_CONTRACT.md` | G1, G5, G6, G7 |
| C++/CUDA hot-path policy | `M1_SCOPE_CONTRACT.md`, `COMPLIANCE_CONTRACT.md` | G5 |
| WAL/checkpoint/replay durability | `M1_SCOPE_CONTRACT.md`, `TEST_GATE_CONTRACT.md` | G2 |
| Performance hard-gate thresholds (Jetson Orin profile) and anti-flake reporting | `TEST_GATE_CONTRACT.md`, `implementationplan.md` | G4 |
| Binary FP32 ingest command and format validation (`bulk-insert-bin`) | `CLI_CONTRACT.md`, `M1_SCOPE_CONTRACT.md` | G1, G3, G4 |
| Contract backward compatibility | `CLI_CONTRACT.md`, `ARTIFACT_CONTRACT.md` | G3 |

## Usage Rule

Before merging implementation changes, update this matrix if:

- A contract section changes, or
- A gate definition changes, or
- A requirement is added/deferred.

