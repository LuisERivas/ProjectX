#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "vector_db/vector_store.hpp"

namespace fs = std::filesystem;

static std::vector<float> make_vec(std::uint64_t id) {
    std::vector<float> v(vector_db_v2::kVectorDim, 0.0f);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<float>((id % 97) * 0.001 + (i % 13) * 0.01);
    }
    return v;
}

static std::optional<std::uint64_t> extract_u64(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = line.find(needle);
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto num_pos = line.find_first_of("0123456789", key_pos + needle.size());
    if (num_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto end_pos = line.find_first_not_of("0123456789", num_pos);
    try {
        return static_cast<std::uint64_t>(std::stoull(line.substr(num_pos, end_pos - num_pos)));
    } catch (...) {
        return std::nullopt;
    }
}

static std::optional<int> extract_i32(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = line.find(needle);
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto num_pos = line.find_first_of("-0123456789", key_pos + needle.size());
    if (num_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto end_pos = line.find_first_not_of("-0123456789", num_pos);
    try {
        return std::stoi(line.substr(num_pos, end_pos - num_pos));
    } catch (...) {
        return std::nullopt;
    }
}

int main() {
    const fs::path data_dir = fs::temp_directory_path() / "vectordb_v2_m1_test_data";
    std::error_code ec;
    fs::remove_all(data_dir, ec);

    vector_db_v2::VectorStore store(data_dir.string());
    auto s = store.init();
    if (!s.ok) {
        std::cerr << "init failed: " << s.message << "\n";
        return 1;
    }
    s = store.open();
    if (!s.ok) {
        std::cerr << "open failed: " << s.message << "\n";
        return 1;
    }

    std::vector<vector_db_v2::Record> batch;
    batch.reserve(200);
    for (std::uint64_t id = 1; id <= 200; ++id) {
        batch.push_back(vector_db_v2::Record{id, make_vec(id)});
    }
    s = store.insert_batch(batch);
    if (!s.ok) {
        std::cerr << "insert_batch failed: " << s.message << "\n";
        return 1;
    }

    const auto wal_before_ckpt = store.wal_stats();
    if (wal_before_ckpt.wal_entries != 200 || wal_before_ckpt.last_lsn != 200) {
        std::cerr << "unexpected wal stats before checkpoint\n";
        return 1;
    }

    s = store.checkpoint();
    if (!s.ok) {
        std::cerr << "checkpoint failed: " << s.message << "\n";
        return 1;
    }
    const auto wal_after_ckpt = store.wal_stats();
    if (wal_after_ckpt.wal_entries != 0 || wal_after_ckpt.checkpoint_lsn != wal_before_ckpt.last_lsn) {
        std::cerr << "unexpected wal stats after checkpoint\n";
        return 1;
    }

    s = store.close();
    if (!s.ok) {
        std::cerr << "close failed: " << s.message << "\n";
        return 1;
    }
    s = store.open();
    if (!s.ok) {
        std::cerr << "re-open failed: " << s.message << "\n";
        return 1;
    }
    const auto st_after_reopen = store.stats();
    if (st_after_reopen.live_rows != 200) {
        std::cerr << "live_rows mismatch after reopen\n";
        return 1;
    }

    s = store.build_top_clusters(1234);
    if (!s.ok) {
        std::cerr << "top failed: " << s.message << "\n";
        return 1;
    }
    s = store.build_mid_layer_clusters(1234);
    if (!s.ok) {
        std::cerr << "mid failed: " << s.message << "\n";
        return 1;
    }
    s = store.build_lower_layer_clusters(1234);
    if (!s.ok) {
        std::cerr << "lower failed: " << s.message << "\n";
        return 1;
    }
    s = store.build_final_layer_clusters(1234);
    if (!s.ok) {
        std::cerr << "final failed: " << s.message << "\n";
        return 1;
    }

    const fs::path root = data_dir / "clusters" / "current";
    const fs::path final_agg = root / "final_layer_clustering" / "FINAL_LAYER_DBSCAN.json";
    const fs::path lower = root / "lower_layer_clustering" / "LOWER_LAYER_CLUSTERING.json";
    const fs::path mid = root / "mid_layer_clustering" / "MID_LAYER_CLUSTERING.json";
    if (!fs::exists(final_agg) || !fs::exists(lower) || !fs::exists(mid)) {
        std::cerr << "required artifacts missing\n";
        return 1;
    }

    bool found_centroid_artifacts = false;
    for (const auto& entry : fs::directory_iterator(root / "final_layer_clustering")) {
        if (!entry.is_directory()) {
            continue;
        }
        const auto dirname = entry.path().filename().string();
        if (dirname.rfind("centroid_", 0) != 0) {
            continue;
        }
        const fs::path labels = entry.path() / "labels.json";
        const fs::path summary = entry.path() / "cluster_summary.json";
        const fs::path manifest = entry.path() / "manifest.json";
        if (!fs::exists(labels) || !fs::exists(summary) || !fs::exists(manifest)) {
            std::cerr << "missing per-centroid final artifacts\n";
            return 1;
        }
        found_centroid_artifacts = true;

        std::ifstream in(labels, std::ios::binary);
        if (!in) {
            std::cerr << "failed opening labels.json\n";
            return 1;
        }
        std::unordered_set<std::uint64_t> seen;
        std::uint64_t prev_id = 0;
        bool first = true;
        std::size_t row_count = 0;
        bool saw_noise = false;
        std::string line;
        while (std::getline(in, line)) {
            const auto id = extract_u64(line, "embedding_id");
            const auto label = extract_i32(line, "label");
            if (!id.has_value() || !label.has_value()) {
                continue;
            }
            if (!first && *id < prev_id) {
                std::cerr << "labels not sorted by embedding_id\n";
                return 1;
            }
            if (seen.find(*id) != seen.end()) {
                std::cerr << "duplicate embedding_id in labels\n";
                return 1;
            }
            if (*label == -1) {
                saw_noise = true;
            }
            seen.insert(*id);
            prev_id = *id;
            first = false;
            ++row_count;
        }
        if (row_count == 0) {
            std::cerr << "labels row_count must be non-zero for written centroid output\n";
            return 1;
        }
        if (!saw_noise) {
            std::cerr << "noise label -1 not found in labels output\n";
            return 1;
        }

        std::ifstream sin(summary, std::ios::binary);
        std::ostringstream sos;
        sos << sin.rdbuf();
        const auto records_processed = extract_u64(sos.str(), "records_processed");
        if (!records_processed.has_value() || static_cast<std::size_t>(*records_processed) != row_count) {
            std::cerr << "labels cardinality mismatch with cluster_summary\n";
            return 1;
        }
    }
    if (!found_centroid_artifacts) {
        std::cerr << "no final centroid artifacts found\n";
        return 1;
    }

    const auto cstats = store.cluster_stats();
    if (!cstats.available) {
        std::cerr << "cluster_stats unavailable\n";
        return 1;
    }
    if (cstats.compliance_status != "pass") {
        std::cerr << "compliance expected pass in default test mode\n";
        return 1;
    }

    const auto health = store.cluster_health();
    if (!health.available || !health.passed) {
        std::cerr << "cluster health not passing\n";
        return 1;
    }

    std::cout << "vectordb_v2_tests: PASS\n";
    return 0;
}
