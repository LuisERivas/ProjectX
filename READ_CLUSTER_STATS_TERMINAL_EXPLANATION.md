# Read Cluster Stats Terminal Output Explanation

This document explains all output shown during the `read cluster stats` stage in `scripts/pipeline_test.py`.  
It covers both the wrapper lines printed by the runner and every JSON field printed by `vectordb_cli cluster-stats`.  
Each item is labeled as either **Flow-Critical**, **Validation-Important**, or **Extra Telemetry**.

## Runner wrapper lines (printed by `pipeline_test.py`)

- `== read cluster stats ==`  
  This is the stage header from the pipeline runner and marks the start of this specific command.  
  It is useful for human readability and log navigation but does not affect clustering logic.  
  **Importance:** Extra Telemetry.

- `$ <path_to_vectordb_cli> cluster-stats --path <data_dir>`  
  This is the exact command being executed so you can copy and rerun it manually for debugging.  
  It confirms which data directory is being queried and which binary was used.  
  **Importance:** Validation-Important.

- `Estimated duration: ~...` and progress bar/ETA line  
  This is a runtime estimate from the script timing model and is only for user feedback.  
  It does not modify state or change how cluster stats are computed.  
  **Importance:** Extra Telemetry.

## `cluster-stats` JSON fields (printed by `vectordb_cli`)

- `available`  
  Indicates whether cluster metadata exists and was loaded successfully.  
  If this is false, other clustering values may be defaults and not meaningful.  
  **Importance:** Flow-Critical.

- `version`  
  The first-layer clustering version currently active in the manifest.  
  This version is used downstream to locate second-level artifacts under `v<version>`.  
  **Importance:** Flow-Critical.

- `build_lsn`  
  The WAL/log sequence number at which the cluster build was recorded.  
  It ties cluster state to a specific durability/log checkpoint position.  
  **Importance:** Validation-Important.

- `vectors_indexed`  
  Number of vectors included in the clustering build.  
  This helps verify expected dataset size participation for the layer.  
  **Importance:** Validation-Important.

- `chosen_k`  
  Final selected number of clusters for first-layer clustering.  
  This is a key model output and drives interpretation of top-layer partitioning.  
  **Importance:** Flow-Critical.

- `k_min`  
  Lower bound of candidate `k` from intrinsic dimensionality estimation.  
  It explains search space constraints but is not itself the final answer.  
  **Importance:** Validation-Important.

- `k_max`  
  Upper bound of candidate `k` from intrinsic dimensionality estimation.  
  Together with `k_min`, it contextualizes why elbow searched a given range.  
  **Importance:** Validation-Important.

- `objective`  
  Final objective value for the selected clustering model.  
  Useful for quality/performance comparison across runs and seeds.  
  **Importance:** Validation-Important.

- `used_cuda`  
  Whether CUDA kernels were used during clustering scoring/reduction stages.  
  In your current GPU-first setup, this should be true on successful GPU runs.  
  **Importance:** Flow-Critical.

- `tensor_core_enabled`  
  Whether tensor-core-capable backend path was engaged.  
  This is an acceleration signal and may vary by hardware/shape/runtime.  
  **Importance:** Validation-Important.

- `gpu_backend`  
  Backend string (for example `cublaslt`) describing the scoring engine used.  
  It helps verify the expected GPU path was actually selected.  
  **Importance:** Validation-Important.

- `scoring_ms_total`  
  Total measured scoring time across scoring calls in the run.  
  This is performance telemetry and not used for control flow.  
  **Importance:** Extra Telemetry.

- `scoring_calls`  
  Number of scoring invocations accumulated in the build.  
  Useful for profiling or comparing search/pruning behavior.  
  **Importance:** Extra Telemetry.

- `elbow_k_evaluated_count`  
  Count of `k` candidates effectively evaluated by elbow logic.  
  Helps confirm pruning/trace behavior and run complexity.  
  **Importance:** Extra Telemetry.

- `elbow_stage_a_candidates`  
  Candidate count considered in stage A of elbow process.  
  Mostly diagnostic for staged elbow behavior.  
  **Importance:** Extra Telemetry.

- `elbow_stage_b_candidates`  
  Candidate count considered in stage B/refined elbow process.  
  Useful for understanding prune-window effects.  
  **Importance:** Extra Telemetry.

- `elbow_early_stop_reason`  
  Text reason if elbow logic stopped early or converged.  
  This is explanatory and valuable for debugging convergence behavior.  
  **Importance:** Extra Telemetry.

- `stability_runs_executed`  
  Number of stability runs actually executed.  
  Confirms whether adaptive stability shortened or extended evaluation.  
  **Importance:** Extra Telemetry.

- `load_live_vectors_ms`  
  Time spent loading live vectors for clustering.  
  This is latency telemetry for I/O and data preparation.  
  **Importance:** Extra Telemetry.

- `id_estimation_ms`  
  Time spent on intrinsic dimensionality estimation step.  
  Useful for timing decomposition and optimization priorities.  
  **Importance:** Extra Telemetry.

- `elbow_ms`  
  Time spent on elbow search and candidate fitting decisions.  
  This is performance telemetry, not a flow gate.  
  **Importance:** Extra Telemetry.

- `stability_ms`  
  Time spent in stability evaluation stage.  
  Helps explain runtime budget across clustering sub-steps.  
  **Importance:** Extra Telemetry.

- `write_artifacts_ms`  
  Time spent writing cluster artifacts to disk.  
  Useful for I/O tuning and end-to-end performance visibility.  
  **Importance:** Extra Telemetry.

- `total_build_ms`  
  Total measured build duration for this clustering pass.  
  Commonly used for top-line runtime comparisons across runs.  
  **Importance:** Validation-Important.

- `live_vector_bytes_read`  
  Number of bytes read while loading vectors for this build.  
  This is I/O telemetry used for throughput diagnostics.  
  **Importance:** Extra Telemetry.

- `live_vector_contiguous_spans`  
  Number of contiguous vector spans detected/read.  
  Indicates how contiguous loading optimizations behaved.  
  **Importance:** Extra Telemetry.

- `live_vector_sparse_reads`  
  Number of sparse row reads performed during vector load.  
  Helps identify fragmented access patterns and fallback behavior.  
  **Importance:** Extra Telemetry.

- `live_vector_sparse_fallback`  
  Whether sparse fallback path was used while loading vectors.  
  Diagnostic flag for data-layout-dependent load behavior.  
  **Importance:** Extra Telemetry.

- `live_vector_async_double_buffer`  
  Whether async double-buffer loading path was used.  
  Useful to confirm asynchronous loading optimization was active.  
  **Importance:** Extra Telemetry.

- `elbow_stage_a_approx_enabled`  
  Indicates if approximate stage-A elbow mode was enabled.  
  This affects exploration strategy and runtime/performance tradeoff.  
  **Importance:** Extra Telemetry.

- `elbow_stage_a_approx_dim`  
  Effective dimensionality used in stage-A approximation mode.  
  Primarily diagnostic when approximate mode is active.  
  **Importance:** Extra Telemetry.

- `elbow_stage_a_approx_stride`  
  Stride used for stage-A approximation downsampling.  
  Helps interpret approximate elbow behavior and speed gains.  
  **Importance:** Extra Telemetry.

- `elbow_stage_b_pruned_candidates`  
  Number of candidates remaining/considered after pruning window logic.  
  Diagnostic metric for candidate-space reduction effectiveness.  
  **Importance:** Extra Telemetry.

- `elbow_stage_b_window_k_min`  
  Lower bound of stage-B pruned window.  
  Useful when debugging why certain `k` values were or were not evaluated.  
  **Importance:** Extra Telemetry.

- `elbow_stage_b_window_k_max`  
  Upper bound of stage-B pruned window.  
  Complements window min for full search-window observability.  
  **Importance:** Extra Telemetry.

- `elbow_stage_b_prune_reason`  
  Reason string for stage-B prune-window selection.  
  Human-readable context for pruning decisions.  
  **Importance:** Extra Telemetry.

- `elbow_int8_search_enabled`  
  Whether INT8 search path was enabled for elbow evaluations.  
  Important for confirming intended precision strategy is active.  
  **Importance:** Validation-Important.

- `elbow_int8_tensor_core_used`  
  Whether tensor-core path was used in INT8 elbow scoring.  
  Useful for hardware-path validation and performance diagnostics.  
  **Importance:** Validation-Important.

- `elbow_int8_eval_count`  
  Number of INT8-based evaluations performed during elbow stage(s).  
  Helps quantify how much of elbow search used INT8 acceleration.  
  **Importance:** Extra Telemetry.

- `elbow_int8_scale_mode`  
  Quantization scaling mode used for INT8 scoring.  
  Important for reproducibility and precision/performance interpretation.  
  **Importance:** Extra Telemetry.

- `elbow_scoring_precision`  
  Effective scoring precision mode summary (for example `fp16` or `int8-search/fp16-final`).  
  This is one of the easiest sanity checks for expected precision policy.  
  **Importance:** Validation-Important.

## Quick practical guidance

- If you are debugging pipeline correctness, focus first on: `available`, `version`, `chosen_k`, and `used_cuda`.  
- If you are validating expected GPU policy, also check: `tensor_core_enabled`, `gpu_backend`, `elbow_int8_search_enabled`, and `elbow_scoring_precision`.  
- Treat the rest as useful profiling context unless a specific performance or convergence issue is being investigated.
