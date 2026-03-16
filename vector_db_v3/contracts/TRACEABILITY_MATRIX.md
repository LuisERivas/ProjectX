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
| Terminal JSONL lifecycle events | `TERMINAL_EVENT_CONTRACT.md` | G7 |
| Active pipeline terminal reporting and previous-run timing baseline | `TERMINAL_EVENT_CONTRACT.md` | G7 |
| CUDA/Tensor/Ampere compliance | `COMPLIANCE_CONTRACT.md` | G5, G6 |
| C++/CUDA hot-path policy | `M1_SCOPE_CONTRACT.md`, `COMPLIANCE_CONTRACT.md` | G5 |
| WAL/checkpoint/replay durability | `M1_SCOPE_CONTRACT.md`, `TEST_GATE_CONTRACT.md` | G2 |
| Performance hard-gate thresholds (Jetson Orin profile) and anti-flake reporting | `TEST_GATE_CONTRACT.md`, `implementationplan.md` | G4 |
| Contract backward compatibility | `CLI_CONTRACT.md`, `ARTIFACT_CONTRACT.md` | G3 |

## Usage Rule

Before merging implementation changes, update this matrix if:

- A contract section changes, or
- A gate definition changes, or
- A requirement is added/deferred.

