# Compliance Contract

## Purpose

Define runtime compliance requirements and machine-verifiable evidence for M1.

## Required Runtime Policy

- Performance-critical stages are CUDA-required.
- Target architecture class is Ampere-capable CUDA deployment.
- Tensor Core use is required where kernels are eligible.
- Performance-critical hot path implementation is C++/CUDA.

## CUDA Kernel and Runtime Best-Practice Requirements

The following requirements are mandatory for performance-critical CUDA kernels and stage orchestration in M1:

- Parallelism:
  - Sequential host/device work in critical path must be parallelized where correctness allows.
  - Long serial loops in critical path require explicit justification in implementation notes.
- Host-device transfers:
  - Host<->device transfers must be minimized.
  - Critical stages should keep reusable working sets resident on device when feasible.
- Launch configuration:
  - Kernel launch configuration (grid/block sizing) must be tuned for high device utilization on target hardware.
  - Fixed launch parameters without utilization rationale are non-compliant for critical kernels.
- Global memory access:
  - Global memory accesses should be coalesced for critical kernels.
  - Repeated redundant global reads/writes should be reduced using reuse strategies where feasible.
- Warp execution behavior:
  - Long diverged execution paths within a warp should be avoided in hot kernels.
  - Branch-heavy kernels in critical path require divergence-aware design or documented mitigation.

Telemetry/verification expectation:

- Compliance evidence must be machine-verifiable in stage outputs/terminal events.
- If critical kernels violate required compliance policy and no documented exception applies, stage must fail with explicit reason.

## Required Compliance Fields

- `cuda_required` (bool)
- `cuda_enabled` (bool)
- `tensor_core_required` (bool)
- `tensor_core_active` (bool)
- `gpu_arch_class` (string)
- `kernel_backend_path` (string)
- `hot_path_language` (string; expected `cpp_cuda`)
- `compliance_status` (`pass` or `fail`)
- `fallback_reason` (string; required when fail)
- `non_compliance_stage` (string; required when fail)

## Compliance Semantics

- `compliance_status=fail` in required stage is terminal for that stage.
- Silent downgrade to non-compliant execution is disallowed.
- Non-critical tooling can be exempt only when explicitly documented.

## Fail-Fast Behavior

When required compliance conditions are not met:

1. Emit `stage_fail` event with compliance details.
2. Return non-zero command status.
3. Persist failure reason in stage outputs where applicable.

