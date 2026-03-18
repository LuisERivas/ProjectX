# Optimization Implementation Playbook: `vector_db_v3`

## Purpose

This document is a copy/paste-ready optimization guide you can hand to a planning agent to create implementation plans for maximum throughput and latency reduction across the full pipeline:

- synthetic embedding generation
- ingest (`bulk-insert` / `bulk-insert-bin`)
- top/mid/lower/final clustering
- terminal summary + result output

It is written for an Ampere-first direction (CUDA cores + Tensor Cores) while preserving existing M1 contracts.

---

## Scope Lock (Do Not Break Existing Behavior)

- Keep contract behavior stable unless explicitly approved:
  - `contracts/M1_SCOPE_CONTRACT.md`
  - `contracts/CLI_CONTRACT.md`
  - `contracts/ARTIFACT_CONTRACT.md`
  - `contracts/BINARY_FORMATS.md`
  - `contracts/TERMINAL_EVENT_CONTRACT.md`
  - `contracts/COMPLIANCE_CONTRACT.md`
  - `contracts/TEST_GATE_CONTRACT.md`
  - `contracts/TRACEABILITY_MATRIX.md`
- Preserve deterministic ordering/outputs where contracts require it.
- Keep existing gate coverage (G1..G7) and extend tests additively.
- Prefer additive CLI/runtime changes before changing existing command semantics.

---

## Baseline Assumption for Estimates

Use this assumption when planning rough savings:

- Example baseline full run: `pipeline_test.py --run-full-pipeline` at 10,000 embeddings takes about 120 seconds.
- All savings below are approximate and overlap (not strictly additive).
- Re-measure on target hardware after each phase.

---

## Prioritized Optimization Backlog (Top 10)

1. GPU k-means kernels (assignment/update) for top/mid clustering.
2. FP16 Tensor Core compute path for distance-heavy k-means math.
3. Keep vectors resident on GPU across pipeline stages.
4. Add internal FP16/INT8 binary embedding shards for clustering path.
5. Async ingest pipeline (pinned memory + chunked async H2D transfers).
6. Single-process full pipeline command (avoid per-stage process reopen).
7. WAL group commit in bulk ingest.
8. Checkpoint immediately after ingest to minimize replay overhead.
9. Stream binary ingest directly to chunked insert (no full materialization).
10. Reduce metadata overhead (manifest payload format + parse cost + readback churn).

---

## Optimization Card 1: CUDA K-Means Core

### Objective
Move the k-means hot loop from CPU to CUDA kernels for assignment and centroid update, preserving current k-selection and no-empty-cluster behavior.

### Why It Matters
- Current `run_deterministic_kmeans` in `src/vector_store.cpp` is CPU scalar over 1024D vectors.
- This is the dominant compute path in top and mid stages.

### Approximate Savings
- ~25s to 60s saved on a 120s baseline (dataset and occupancy dependent).

### Primary Files
- `src/vector_store.cpp`
- New CUDA source(s), for example:
  - `src/cuda/kmeans_kernels.cu`
  - `src/cuda/kmeans_runtime.cpp`
- `include/vector_db_v3/vector_store.hpp` (only if API seam needed)
- `CMakeLists.txt` (CUDA target wiring)

### Implementation Notes
- Keep existing CPU path as fallback behind runtime/compile switch.
- Use deterministic tie-break rules consistent with current outputs.
- Maintain empty-cluster repair semantics.

### Risks
- Determinism drift.
- Numeric drift vs CPU path.
- Build/toolchain complexity in CI.

### Acceptance Criteria
- G1/G3/G7 remain green.
- Cluster assignments/manifests remain contract-valid.
- Performance evidence shows measurable stage latency reduction.

### Copy/Paste Prompt For Planning Agent
```text
Create an execution-ready plan to add a CUDA k-means backend for vector_db_v3 while preserving current contracts and deterministic behavior.
Scope:
- Keep current CPU implementation as fallback.
- Implement assignment and centroid update kernels.
- Integrate backend selection into top/mid stage execution.
- Ensure no-empty-cluster semantics are preserved.
- Add tests for CPU-vs-CUDA parity tolerances and deterministic tie behavior.
Files to target:
- src/vector_store.cpp
- src/cuda/kmeans_kernels.cu (new)
- src/cuda/kmeans_runtime.cpp (new)
- CMakeLists.txt
Output:
- phased plan, risks, test strategy, perf validation commands, rollback path.
```

---

## Optimization Card 2: Tensor Core FP16 Distance Path

### Objective
Implement an FP16 Tensor Core path for distance/score computations in clustering (for Ampere), while preserving correctness and ranking semantics.

### Why It Matters
- Ampere Tensor Cores can accelerate matrix-like operations heavily vs scalar FP32 loops.
- Especially useful for assignment distance calculations.

### Approximate Savings
- ~15s to 40s saved on a 120s baseline.

### Primary Files
- `src/vector_store.cpp` (backend invocation)
- New compute/runtime files:
  - `src/cuda/tensor_distance.cu`
  - `src/cuda/tensor_runtime.cpp`
- `CMakeLists.txt`

### Implementation Notes
- Keep FP32 boundary where contract requires.
- Use FP16 compute and accumulate strategy with controlled tolerance policy.
- Runtime gating by hardware capability and compliance checks.

### Risks
- Numerical stability and cluster boundary changes.
- Hardware-specific tuning requirements.

### Acceptance Criteria
- Contract tests pass.
- Reproducibility checks stable within declared tolerance.
- Perf gate shows lower median and p95 stage times.

### Copy/Paste Prompt For Planning Agent
```text
Plan Tensor Core acceleration for clustering distance computations in vector_db_v3.
Requirements:
- Add FP16 Tensor Core execution path for Ampere.
- Preserve contract-visible behavior and deterministic ordering.
- Define tolerance policy and parity tests against CPU baseline.
- Add runtime compliance gating and graceful fallback.
Files:
- src/vector_store.cpp
- src/cuda/tensor_distance.cu (new)
- src/cuda/tensor_runtime.cpp (new)
- CMakeLists.txt
Deliver:
- implementation phases, test updates, perf measurement plan, risk mitigation.
```

---

## Optimization Card 3: GPU Residency Across Stages

### Objective
Keep embedding and working buffers in GPU memory across top->mid->lower->final to minimize repeated host-device transfers.

### Why It Matters
- Stage-to-stage transfer churn can erase CUDA gains.
- Residency allows reusing transformed/packed buffers.

### Approximate Savings
- ~10s to 30s saved on a 120s baseline.

### Primary Files
- `src/vector_store.cpp`
- New runtime/context manager file:
  - `src/cuda/pipeline_context.cpp`
  - `include/vector_db_v3/cuda_pipeline_context.hpp`

### Implementation Notes
- Add lifecycle-managed GPU context owned by process/store instance.
- Keep deterministic stage boundaries and telemetry unchanged.

### Risks
- Memory pressure on device.
- Cleanup/error-path complexity.

### Acceptance Criteria
- No contract drift.
- Stable long-run memory behavior.
- Reduced stage elapsed times in telemetry.

### Copy/Paste Prompt For Planning Agent
```text
Create a plan to introduce GPU buffer residency across clustering stages in vector_db_v3.
Goal:
- Avoid repeated host-device copies by keeping stage inputs/outputs resident.
Constraints:
- Preserve terminal eventing, stage semantics, and artifact contracts.
- Define fallback when GPU memory is insufficient.
Files:
- src/vector_store.cpp
- src/cuda/pipeline_context.cpp (new)
- include/vector_db_v3/cuda_pipeline_context.hpp (new)
Include:
- lifecycle design, failure handling, perf validation, rollback strategy.
```

---

## Optimization Card 4: Faster Internal Embedding Formats (FP16/INT8 Shards)

### Objective
Add additive internal binary shards for FP16/INT8 embeddings to reduce bandwidth and memory pressure for clustering and candidate processing.

### Why It Matters
- FP16 is 2x smaller than FP32.
- INT8 is 4x smaller than FP32 for bandwidth-sensitive paths.

### Approximate Savings
- ~8s to 25s saved on a 120s baseline (depends on where conversion is amortized).

### Primary Files
- `src/vector_store.cpp`
- `src/codec/artifacts.cpp`
- `src/codec/types.hpp`
- `contracts/*` only if new external artifacts become contract-visible

### Implementation Notes
- Keep existing FP32 ingest boundary unchanged.
- Produce internal shard files additively and ID-aligned.
- Avoid mandatory contract changes unless artifacts are externally required.

### Risks
- Conversion overhead if done repeatedly.
- Precision drift if used beyond intended scope.

### Acceptance Criteria
- Existing artifacts and commands unchanged.
- ID-alignment consistency checks remain valid.
- Net latency improvement after conversion overhead.

### Copy/Paste Prompt For Planning Agent
```text
Plan additive internal FP16/INT8 embedding shard support in vector_db_v3 for throughput optimization.
Requirements:
- Keep FP32 ingest and existing contracts stable.
- Add ID-aligned internal shard generation and reuse.
- Define when to read FP32 vs FP16 vs INT8 paths.
- Include validation for numeric consistency and fallback logic.
Files:
- src/vector_store.cpp
- src/codec/artifacts.cpp
- src/codec/types.hpp
Output:
- phased implementation with minimal contract impact and perf measurement plan.
```

---

## Optimization Card 5: Async Ingest Pipeline (Pinned + Streamed)

### Objective
Overlap file read, decode, transfer, and compute using chunked ingest with pinned host memory and async streams.

### Why It Matters
- Reduces idle gaps between I/O and compute.
- Better feed rate into GPU path.

### Approximate Savings
- ~6s to 20s saved on a 120s baseline.

### Primary Files
- `cli/main.cpp`
- `src/vector_store.cpp`
- optional new helper:
  - `src/cuda/ingest_pipeline.cpp`

### Implementation Notes
- Keep current CLI output contract.
- Ensure chunk ordering and error semantics remain deterministic.

### Risks
- Concurrency bugs and harder debugging.
- Platform differences for pinned memory behavior.

### Acceptance Criteria
- Same inserted counts and command payload semantics.
- Improved ingest stage latency under perf gate.

### Copy/Paste Prompt For Planning Agent
```text
Create a plan to implement an async chunked ingest pipeline for vector_db_v3.
Scope:
- Overlap file read/decode with transfer/processing.
- Use pinned host memory and stream-based execution where available.
- Preserve existing CLI response schema and fail semantics.
Files:
- cli/main.cpp
- src/vector_store.cpp
- src/cuda/ingest_pipeline.cpp (new, optional)
Deliverables:
- design, staged rollout, correctness checks, perf validation matrix.
```

---

## Optimization Card 6: Single-Process Full Pipeline Command

### Objective
Add a composite CLI command to execute all stages in one process, preserving current stage boundaries and events.

### Why It Matters
- Current script runs multiple subprocesses, each with open/replay overhead.
- One process enables GPU context reuse and less startup churn.

### Approximate Savings
- ~8s to 20s saved on a 120s baseline.

### Primary Files
- `cli/main.cpp`
- `src/vector_store.cpp`
- `scripts/pipeline_test.py` (optional switch to new command)

### Implementation Notes
- Keep existing individual commands unchanged.
- Composite command should be additive.

### Risks
- Mixed concerns between stage orchestration and command dispatch.

### Acceptance Criteria
- Stage-level telemetry/events still emitted per contract.
- Existing commands unaffected.

### Copy/Paste Prompt For Planning Agent
```text
Plan an additive single-process full pipeline CLI command for vector_db_v3.
Requirements:
- Execute init/ingest/top/mid/lower/final in one process.
- Preserve per-stage telemetry and existing command behavior.
- Keep old commands unchanged and available.
Files:
- cli/main.cpp
- src/vector_store.cpp
- scripts/pipeline_test.py (optional use of new command)
Include:
- compatibility strategy, test updates, rollout/rollback plan.
```

---

## Optimization Card 7: WAL Group Commit For Bulk Insert

### Objective
Replace per-record WAL fsync behavior in bulk paths with configurable per-batch commit.

### Why It Matters
- Current path appends + flushes + syncs per record.
- This is expensive and limits ingest throughput.

### Approximate Savings
- ~15s to 40s saved on a 120s baseline.

### Primary Files
- `src/vector_store.cpp`
- possibly `include/vector_db_v3/vector_store.hpp` for config exposure
- tests for durability expectations

### Implementation Notes
- Keep strict mode available.
- Make policy explicit and testable.
- Preserve correctness/recovery guarantees for selected mode.

### Risks
- Durability window trade-offs.
- Recovery behavior differences if misconfigured.

### Acceptance Criteria
- Durability tests updated for configured mode.
- Replay/checkpoint correctness preserved.
- Ingest stage latency reduced.

### Copy/Paste Prompt For Planning Agent
```text
Create an implementation plan for WAL group commit in vector_db_v3 bulk insert paths.
Scope:
- Add configurable commit policy: strict per-record vs batched commit.
- Preserve crash-recovery correctness and explicit mode semantics.
- Keep existing CLI behavior stable by default unless approved.
Files:
- src/vector_store.cpp
- durability tests under tests/
Output:
- durability risk analysis, test matrix, migration plan, perf targets.
```

---

## Optimization Card 8: Post-Ingest Checkpoint Shortcut

### Objective
Run checkpoint after ingest in pipeline workflows to reduce repeated WAL replay costs.

### Why It Matters
- Stage commands currently reopen and replay WAL.
- Checkpoint trims replay burden before clustering stages.

### Approximate Savings
- ~8s to 25s saved on a 120s baseline.

### Primary Files
- `scripts/pipeline_test.py`
- `scripts/run_full_pipeline.py`
- optional CLI orchestration in `cli/main.cpp`

### Implementation Notes
- Use existing `checkpoint` command capability.
- Keep this optional/configurable for benchmark fairness.

### Risks
- Extra checkpoint I/O if dataset is small.

### Acceptance Criteria
- No behavior regressions.
- Measurable reduction in top/mid stage start latency.

### Copy/Paste Prompt For Planning Agent
```text
Plan a post-ingest checkpoint optimization for vector_db_v3 pipeline workflows.
Requirements:
- Add optional checkpoint immediately after successful ingest.
- Preserve result payloads and stage sequencing.
- Measure replay/startup latency reduction in subsequent stages.
Files:
- scripts/pipeline_test.py
- scripts/run_full_pipeline.py
- optional cli/main.cpp integration
Deliver:
- toggle design, validation strategy, performance evidence requirements.
```

---

## Optimization Card 9: Stream `bulk-insert-bin` Without Full Materialization

### Objective
Parse binary records in streaming chunks directly into `insert_batch`, avoiding full in-memory `records` vector.

### Why It Matters
- Reduces memory pressure and copy overhead.
- Improves ingest latency and scalability.

### Approximate Savings
- ~3s to 12s saved on a 120s baseline.

### Primary Files
- `cli/main.cpp`

### Implementation Notes
- Keep header validation unchanged.
- Reuse current chunk/batch behavior and payload output format.

### Risks
- Partial-read edge cases.

### Acceptance Criteria
- `bulk-insert-bin` contract tests remain green.
- Lower memory footprint and improved ingest timing.

### Copy/Paste Prompt For Planning Agent
```text
Create a plan to refactor bulk-insert-bin in vector_db_v3 to streaming chunked ingest.
Scope:
- Keep binary header validation and command output schema unchanged.
- Remove full in-memory record materialization.
- Insert by chunk directly to store.insert_batch.
File:
- cli/main.cpp
Include:
- edge-case handling plan, regression tests, and performance checkpoints.
```

---

## Optimization Card 10: Reduce Metadata/Manifest Overhead

### Objective
Lower non-compute overhead from manifest payload building/parsing and file readback for checksums.

### Why It Matters
- Multiple stages currently read back files and parse JSON payload strings/regex.
- This adds avoidable I/O and CPU overhead.

### Approximate Savings
- ~1s to 8s saved on a 120s baseline.

### Primary Files
- `src/vector_store.cpp`
- `src/codec/artifacts.cpp`
- `scripts/pipeline_test.py`

### Implementation Notes
- Prefer computing checksums from in-memory buffers at write-time.
- Use structured/binary internal summaries where possible.
- Keep external contract outputs unchanged unless approved.

### Risks
- Contract drift if format changes leak externally.

### Acceptance Criteria
- Same observable payload/summary fields where required.
- Reduced per-stage metadata overhead in profiling.

### Copy/Paste Prompt For Planning Agent
```text
Plan a metadata overhead reduction pass for vector_db_v3 stage artifacts and summaries.
Goals:
- minimize file readback after writes,
- reduce expensive payload parsing paths,
- keep external contract-visible outputs stable.
Files:
- src/vector_store.cpp
- src/codec/artifacts.cpp
- scripts/pipeline_test.py
Deliver:
- low-risk optimization sequence, regression guardrails, and measurable KPIs.
```

---

## Recommended Implementation Order (Max ROI + Low Risk First)

### Phase 1 (Low Risk, Immediate Wins)
1. Card 9 (stream binary ingest)
2. Card 8 (post-ingest checkpoint)
3. Card 7 (WAL group commit, conservative mode)
4. Card 6 (single-process pipeline command)

### Phase 2 (Medium/High Impact, Medium Risk)
5. Card 10 (metadata overhead reduction)
6. Card 4 (internal FP16/INT8 shards)
7. Card 5 (async ingest pipeline)

### Phase 3 (Highest Compute Gain, Highest Complexity)
8. Card 1 (CUDA k-means)
9. Card 2 (Tensor Core FP16 path)
10. Card 3 (GPU residency across stages)

---

## Global Verification Checklist (Use After Every Card)

- G1 correctness tests pass.
- G3 contract stability tests pass.
- G7 terminal event contract tests pass.
- Reproducibility checks remain acceptable.
- `pipeline_test.py` full run still produces:
  - valid stage lifecycle output
  - valid cluster summary
  - valid results JSON
- Perf evidence updated with median/p95 comparison against baseline.

---

## Standard Evidence Pack Template

For each optimization card, require:

- `before/` and `after/` performance runs (same dataset + env controls)
- summary JSON with:
  - stage latencies
  - total latency
  - median and p95 over repeated runs
  - pass/fail against threshold policy
- notes on:
  - behavior deltas
  - risk findings
  - rollback decision

---

## One-Shot Master Prompt (Optional)

Use this if you want one planning task to break all cards into executable work:

```text
You are in PLAN MODE only. Create an execution-ready phased implementation plan for vector_db_v3 optimizationimplementation.md.
Requirements:
- Build plans for all 10 optimization cards.
- Preserve all current contract-visible behavior unless explicitly marked additive.
- Prioritize by ROI and risk using the phase order in the document.
- For each card include: file touch list, design approach, risks, gate impact (G1/G3/G7), validation commands, acceptance criteria, and rollback.
- Add measurable KPIs and expected time savings windows per card.
- Include cross-card dependency graph and anti-regression strategy.
Output as a phase-by-phase plan with clear stop/go criteria.
```

