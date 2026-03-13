# Vector DB v2 M1 Traceability Matrix

This matrix maps `vector_db_v2` source-of-truth docs to implementation targets under `vector_db_v2/` code.

| Doc Clause ID | Requirement Summary | Target Code Path | Implementation Status | Tests | Telemetry/Contract Field | Risk |
|---|---|---|---|---|---|---|
| 01-M1-EXACT | Exact-only retrieval/ranking | `vector_db_v2/src/vector_store.cpp`, `vector_db_v2/cli/main.cpp` | Implemented | `vector_db_v2/tests/test_m1.cpp` + pipeline script | `search` output `{embedding_id, score}` | Similarity metric is cosine baseline only |
| 01/05-HIER-4L | Top -> Mid -> Lower -> Final pipeline | `vector_db_v2/src/vector_store.cpp` build stage methods | Implemented | `test_m1.cpp`, `scripts/pipeline_test.py` | stage lifecycle events + layer summaries | Lower split policy is deterministic heuristic |
| 05-LOWER-GATE | Lower split gate (`continue`/`stop`) semantics | `write_lower_artifacts()` | Implemented | pipeline + artifact checks | `gate_decision` in lower summary/events | Subgroup geometry is simplified for M1 bootstrap |
| 05/06-FINAL-ELIG | Final eligibility: gate `stop` + preflight pass | `write_final_artifacts()`, `check_preflight()` | Implemented | pipeline + final artifact checks | `final_layer_eligibility_reason`, `final_layer_preflight_valid` | Preflight reason enum can be expanded |
| 05/06/07-LABELS | Canonical `labels.json` schema + ordering + uniqueness + noise `-1` | `run_fake_dbscan_labels()`, labels writer | Implemented | `test_m1.cpp` + planned schema checks | `labels_file_present`, `labels_schema_valid` | Runtime schema validator can be strengthened |
| 03-STORAGE | `clusters/current/` active layout + required artifacts | `write_top_artifacts()`, mid/lower/final writers | Implemented | `test_m1.cpp` path assertions | Artifact manifest + summary JSON | Legacy layout intentionally not supported |
| 03-DURABILITY | WAL/checkpoint + atomic file replacement | `append_wal()`, `checkpoint()`, atomic write helpers | Implemented | `test_m1.cpp` + pipeline run | `wal-stats` contract fields | WAL replay is minimal in this cut |
| 04/06-CUDA | CUDA-required, Tensor-core-first contract telemetry + fail-fast support | `apply_compliance()`, stage builders | Implemented | default tests + env-forced non-compliance path | `compliance_status`, `fallback_reason`, `non_compliance_stage` | Uses compliance contract simulation by default |
| 06-TERMINAL-JSONL | Machine-parseable stage lifecycle/timing output | `emit_event()` + stage builder calls | Implemented | pipeline run log inspection | `stage_start`, `stage_end`, `stage_fail`, `pipeline_summary` | Timestamps currently epoch-seconds style string |
| 06-CLI-SURFACE | Strict v2 command surface | `vector_db_v2/cli/main.cpp` | Implemented | `pipeline_test.py` | build-* commands + stats/health | Optional commands not in M1 intentionally omitted |
| 07-GATES | Contract/test gates for artifacts + sequencing | `vector_db_v2/tests/test_m1.cpp`, `scripts/pipeline_test.py` | In progress | build + ctest + pipeline script | phase reports + JSON outputs | Additional negative tests still needed |

## Phase Checkpoint Conventions

- Each implementation phase reports:
  - files changed
  - doc clauses satisfied
  - tests run/results
  - remaining risks/ambiguities
