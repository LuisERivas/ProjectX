# Vector DB v2 Context

This folder is the clean-slate context set for designing and implementing a new Vector DB.
It is docs-only and intentionally separate from current `vector_db/` implementation details.

## Purpose

- Provide a single source of truth for v2 goals, architecture, contracts, and delivery milestones.
- Make design decisions explicit before implementation starts.
- Reduce ambiguity for future coding, testing, and review.

## Inputs/Dependencies

- Existing project constraints in `README.md`, `SETUP.md`, and `TESTING.md`.
- Current v1/v1.5 learnings from `vector_db/` behavior and pipeline orchestration.
- Target deployment assumptions: M1 target is Ampere-class NVIDIA CUDA-capable deployment (including Ampere-class Jetson variants where applicable), plus local filesystem durability.

## Reading Order

1. `01_PRODUCT_AND_REQUIREMENTS.md`
2. `02_SYSTEM_ARCHITECTURE.md`
3. `03_STORAGE_AND_DURABILITY.md`
4. `04_INDEXING_AND_SEARCH.md`
5. `05_CLUSTERING_PIPELINE.md`
6. `06_API_AND_CLI_CONTRACTS.md`
7. `07_TESTING_AND_VALIDATION.md`
8. `08_ROADMAP_AND_MILESTONES.md`
9. `GLOSSARY.md`

## Document Map

- Requirements: [01 Product and Requirements](./01_PRODUCT_AND_REQUIREMENTS.md)
- Architecture: [02 System Architecture](./02_SYSTEM_ARCHITECTURE.md)
- Storage/Durability: [03 Storage and Durability](./03_STORAGE_AND_DURABILITY.md)
- Index/Search: [04 Indexing and Search](./04_INDEXING_AND_SEARCH.md)
- Clustering: [05 Clustering Pipeline](./05_CLUSTERING_PIPELINE.md)
- API/CLI: [06 API and CLI Contracts](./06_API_AND_CLI_CONTRACTS.md)
- Testing: [07 Testing and Validation](./07_TESTING_AND_VALIDATION.md)
- Delivery Plan: [08 Roadmap and Milestones](./08_ROADMAP_AND_MILESTONES.md)
- Terms: [Glossary](./GLOSSARY.md)

## Decision Log Index

Use this index to track major accepted decisions and their alternatives.

- D-001: CUDA-required exact scoring path with explicit hard-fail on missing required hardware/runtime in M1 (see `04`, `05`)
- D-006: M1 numeric boundary is FP32 input embeddings -> INT8 k-selection -> FP16 final clustering/assignment (see `01`, `04`, `05`)
- D-002: Segment + WAL + checkpoint durability model (see `03`)
- D-003: Single active clustering artifact contract in M1; artifact/version history deferred (see `05`, `06`, `08`)
- D-004: Backward-compatible CLI/JSON schema policy for additive evolution (see `06`)
- D-005: Test gates require functional + performance + artifact validation (see `07`)
- D-007: Clustering hierarchy is Top -> Mid -> Lower -> Final, with Final layer executing DBSCAN only for eligible gate-fail (`stop`) Lower-layer centroid datasets that pass required DBSCAN preflight checks; gate `continue` branches remain in Lower processing (no cross-centroid mixing) (see `02`, `05`, `06`)
- D-011: Final-layer per-centroid artifact contract in M1 is minimally normative: `manifest.json`, `labels.json`, `cluster_summary.json` under `final_layer_clustering/centroid_<id>/`, plus aggregate `final_layer_clustering/FINAL_LAYER_DBSCAN.json` (see `03`, `05`, `06`, `07`)
- D-012: Final-layer `labels.json` contract is canonical in M1 (array of `{embedding_id, label}` rows; `label=-1` noise; sorted ascending by `embedding_id`; unique IDs; cardinality equals processed embeddings), and DBSCAN preflight failures require machine-readable reason reporting in events and per-centroid telemetry (see `05`, `06`, `07`, `GLOSSARY`)
- D-008: M1 hardware policy is Ampere-targeted CUDA-required execution with Tensor Cores as a primary compute target across performance-critical stages (see `01`, `04`, `05`, `07`, `08`).
- D-009: M1 hot-path implementation is C++-first (C++/CUDA), and higher-level scripting does not replace core compute kernels (see `01`, `02`, `05`, `08`).
- D-010: M1 terminal stage observability is mandatory by default: machine-parseable stage start/end/fail events with per-stage and cumulative pipeline timing (see `01`, `05`, `06`, `07`, `08`).

## Open Questions

- Should v2 support optional remote object-store snapshots in addition to local filesystem?
- Do we need online compaction in M2, or can we defer to M3 after baseline stability?
- Should query-time hybrid scoring be introduced after M1, and if so in which milestone?

## Exit Criteria

- All linked docs exist and are internally consistent.
- Every major decision lists at least one rejected alternative and rationale.
- The roadmap is concrete enough to start implementation without re-planning fundamentals.
