# ignoreFUTURESTUFF

## Tasks

- [ ] Add deterministic timeout handling in worker execution wrapper (terminalize or DLQ + ACK).
- [ ] Add poison-message handling for schema/envelope failures (DLQ + ACK).
- [ ] Move DLQ stream key default into shared config and reference it from redis key helper.
- [ ] Implement Ampere full-utilization optimization pass for vector DB clustering pipeline: (1) keep k-means iterations fully device-resident in `vector_db/src/clustering/spherical_kmeans_cuda.cu` and `vector_db/src/clustering/elbow_search.cpp` by removing per-iteration host copies/global sync, (2) add stream-based overlap with async H2D/D2H and pinned host buffers while binding cuBLASLt to compute streams, (3) eliminate default full score-matrix D2H by returning compact top-1/top-m/objective outputs and only allowing full score export behind debug gate, (4) optimize atomic-heavy centroid reduction and naive top-m selection kernels with Ampere-friendly shared-memory/warp-level approaches, (5) standardize packed row-major host input path in `vector_db/src/vector_store.cpp` to remove duplicate staging/repacking before GPU kickoff; validate with `scripts/test_vector_db_phase1.py`, `vector_db/tests/smoke_cli.py`, and `vector_db/tests/smoke_cli_profile.py`, preserving current hard-fail GPU policy and current INT8-elbow + FP16-final behavior.
- [ ] db scan for multiple layers based on frequency of access
