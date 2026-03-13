#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vector_db/status.hpp"

namespace vector_db_v2 {

constexpr std::size_t kVectorDim = 1024;

struct Record {
    std::uint64_t embedding_id = 0;
    std::vector<float> vector;
};

struct SearchResult {
    std::uint64_t embedding_id = 0;
    double score = 0.0;
};

struct Stats {
    std::size_t dimension = kVectorDim;
    std::size_t total_rows = 0;
    std::size_t live_rows = 0;
    std::size_t tombstone_rows = 0;
};

struct WalStats {
    std::uint64_t checkpoint_lsn = 0;
    std::uint64_t last_lsn = 0;
    std::size_t wal_entries = 0;
};

struct ClusterStats {
    bool available = false;
    std::uint64_t build_lsn = 0;
    std::size_t vectors_indexed = 0;
    std::size_t chosen_k = 0;
    std::size_t k_min = 0;
    std::size_t k_max = 0;
    double objective = 0.0;

    bool cuda_required = true;
    bool cuda_enabled = false;
    bool tensor_core_required = true;
    bool tensor_core_active = false;
    std::string gpu_arch_class = "unknown";
    std::string kernel_backend_path = "none";
    std::string hot_path_language = "cpp_cuda";
    std::string compliance_status = "fail";
    std::string fallback_reason;
    std::string non_compliance_stage;
};

struct ClusterHealth {
    bool available = false;
    bool passed = true;
    double mean_nmi = 0.0;
    double std_nmi = 0.0;
    double mean_jaccard = 0.0;
    double mean_centroid_drift = 0.0;
    std::string status = "ok";
};

struct BulkInsertMetrics {
    std::size_t rows = 0;
    double wal_ms = 0.0;
    double persist_ms = 0.0;
    double total_ms = 0.0;
};

class VectorStore {
public:
    explicit VectorStore(std::string data_dir);
    ~VectorStore();

    Status init();
    Status open();
    Status close();
    Status checkpoint();

    Status insert(std::uint64_t embedding_id, const std::vector<float>& vector_fp32_1024);
    Status insert_batch(const std::vector<Record>& records);
    Status remove(std::uint64_t embedding_id);
    std::optional<Record> get(std::uint64_t embedding_id) const;
    std::vector<SearchResult> search_exact(const std::vector<float>& query, std::size_t top_k) const;

    Stats stats() const;
    WalStats wal_stats() const;
    ClusterStats cluster_stats() const;
    ClusterHealth cluster_health() const;
    BulkInsertMetrics last_bulk_insert_metrics() const;

    Status build_top_clusters(std::uint32_t seed = 1234);
    Status build_mid_layer_clusters(std::uint32_t seed = 1234);
    Status build_lower_layer_clusters(std::uint32_t seed = 1234);
    Status build_final_layer_clusters(std::uint32_t seed = 1234);

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace vector_db_v2
