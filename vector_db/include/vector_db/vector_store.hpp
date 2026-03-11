#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vector_db/status.hpp"

namespace vector_db {

constexpr std::size_t kVectorDim = 1024;

struct Record {
    std::uint64_t id = 0;
    std::vector<float> vector_fp32;
    std::string metadata_json;
};

struct StoredRecord {
    std::uint64_t id = 0;
    bool deleted = false;
    std::vector<float> vector_fp32;
    std::string metadata_json;
};

struct DirtyRange {
    std::uint64_t segment_id = 1;
    std::size_t start_row = 0;
    std::size_t end_row = 0;
    std::string reason;
};

struct Stats {
    std::size_t dimension = kVectorDim;
    std::size_t total_rows = 0;
    std::size_t live_rows = 0;
    std::size_t tombstone_rows = 0;
    std::size_t segments = 0;
    std::size_t dirty_ranges = 0;
};

struct WalStats {
    std::uint64_t checkpoint_lsn = 0;
    std::uint64_t last_lsn = 0;
    std::size_t wal_entries = 0;
};

struct ClusterStats {
    bool available = false;
    std::uint64_t version = 0;
    std::uint64_t build_lsn = 0;
    std::size_t vectors_indexed = 0;
    std::size_t chosen_k = 0;
    std::size_t k_min = 0;
    std::size_t k_max = 0;
    double objective = 0.0;
    bool used_cuda = false;
    bool tensor_core_enabled = false;
    std::string gpu_backend = "cpu";
    double scoring_ms_total = 0.0;
    std::size_t scoring_calls = 0;
    std::size_t elbow_k_evaluated_count = 0;
    std::size_t elbow_stage_a_candidates = 0;
    std::size_t elbow_stage_b_candidates = 0;
    std::string elbow_early_stop_reason;
    std::size_t stability_runs_executed = 0;
    double load_live_vectors_ms = 0.0;
    double id_estimation_ms = 0.0;
    double elbow_ms = 0.0;
    double stability_ms = 0.0;
    double write_artifacts_ms = 0.0;
    double total_build_ms = 0.0;
    std::size_t live_vector_bytes_read = 0;
    std::size_t live_vector_contiguous_spans = 0;
    std::size_t live_vector_sparse_reads = 0;
    bool live_vector_sparse_fallback = false;
    bool live_vector_async_double_buffer = false;
    bool elbow_stage_a_approx_enabled = false;
    std::size_t elbow_stage_a_approx_dim = 0;
    std::size_t elbow_stage_a_approx_stride = 1;
};

struct ClusterHealth {
    bool available = false;
    bool passed = false;
    double mean_nmi = 0.0;
    double std_nmi = 0.0;
    double mean_jaccard = 0.0;
    double mean_centroid_drift = 0.0;
    std::string status;
};

class VectorStore {
public:
    explicit VectorStore(std::string data_dir);
    ~VectorStore();

    Status init();
    Status open();
    Status flush();
    Status checkpoint();
    Status build_initial_clusters(std::uint32_t seed = 1234);
    Status close();

    Status insert(std::uint64_t id, const std::vector<float>& vector_fp32_1024, const std::string& metadata_json, bool upsert = false);
    Status insert_batch(const std::vector<Record>& records);
    Status remove(std::uint64_t id);
    Status update_metadata(std::uint64_t id, const std::string& patch_json);
    std::optional<StoredRecord> get(std::uint64_t id) const;
    Stats stats() const;
    WalStats wal_stats() const;
    ClusterStats cluster_stats() const;
    ClusterHealth cluster_health() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace vector_db

