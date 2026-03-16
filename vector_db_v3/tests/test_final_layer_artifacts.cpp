#include <filesystem>
#include <iostream>
#include <limits>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include "vector_db_v3/codec/artifacts.hpp"
#include "vector_db_v3/paths.hpp"
#include "vector_db_v3/vector_store.hpp"

namespace fs = std::filesystem;

namespace {

bool expect(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        return false;
    }
    return true;
}

std::vector<float> make_vec(float base) {
    std::vector<float> v(vector_db_v3::kVectorDim, 0.0f);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = base + static_cast<float>((i % 11U) * 0.0015f);
    }
    return v;
}

std::uint32_t parse_u32_or_default(const std::string& body, const std::string& key, std::uint32_t fallback) {
    const std::regex re("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch m;
    if (!std::regex_search(body, m, re) || m.size() < 2) {
        return fallback;
    }
    try {
        return static_cast<std::uint32_t>(std::stoul(m[1].str()));
    } catch (...) {
        return fallback;
    }
}

std::set<std::uint32_t> parse_cluster_ids(const std::string& payload) {
    std::set<std::uint32_t> ids;
    const std::regex id_re("\"final_cluster_numeric_id\"\\s*:\\s*([0-9]+)");
    auto it = std::sregex_iterator(payload.begin(), payload.end(), id_re);
    const auto end = std::sregex_iterator();
    for (; it != end; ++it) {
        const std::smatch& m = *it;
        if (m.size() < 2) {
            continue;
        }
        ids.insert(static_cast<std::uint32_t>(std::stoul(m[1].str())));
    }
    return ids;
}

}  // namespace

int main() {
    const fs::path data_dir = fs::temp_directory_path() / "vectordb_v3_final_layer_artifacts_test";
    std::error_code ec;
    fs::remove_all(data_dir, ec);

    bool ok = true;
    vector_db_v3::VectorStore store(data_dir.string());
    ok &= expect(store.init().ok, "init");
    ok &= expect(store.open().ok, "open");
    for (std::uint64_t id = 1; id <= 64; ++id) {
        ok &= expect(store.insert(id, make_vec(static_cast<float>(id) * 0.05f)).ok, "insert");
    }

    ok &= expect(store.build_top_clusters(9U).ok, "build_top_clusters");
    ok &= expect(store.build_mid_layer_clusters(9U).ok, "build_mid_layer_clusters");
    ok &= expect(store.build_lower_layer_clusters(9U).ok, "build_lower_layer_clusters");
    ok &= expect(store.build_final_layer_clusters(9U).ok, "build_final_layer_clusters");

    const fs::path final_clusters_bin = vector_db_v3::paths::final_layer_clusters_bin(data_dir);
    const fs::path k_search_bin = vector_db_v3::paths::k_search_bounds_batch_bin(data_dir);
    const fs::path post_membership_bin = vector_db_v3::paths::post_cluster_membership_bin(data_dir);
    ok &= expect(fs::exists(final_clusters_bin), "FINAL_LAYER_CLUSTERS.bin exists");
    ok &= expect(fs::exists(k_search_bin), "k_search_bounds_batch.bin exists");
    ok &= expect(fs::exists(post_membership_bin), "post_cluster_membership.bin exists");

    vector_db_v3::codec::CommonHeader aggregate_hdr{};
    std::vector<std::uint8_t> aggregate_payload_bytes;
    ok &= expect(
        vector_db_v3::codec::read_cluster_manifest_file(final_clusters_bin, &aggregate_hdr, &aggregate_payload_bytes).ok,
        "read FINAL_LAYER_CLUSTERS.bin");
    const std::string aggregate_payload(aggregate_payload_bytes.begin(), aggregate_payload_bytes.end());
    const std::set<std::uint32_t> cluster_ids = parse_cluster_ids(aggregate_payload);
    for (const std::uint32_t cluster_id : cluster_ids) {
        const fs::path manifest_path = vector_db_v3::paths::final_cluster_manifest_bin(data_dir, cluster_id);
        const fs::path assignments_path = vector_db_v3::paths::final_cluster_assignments_bin(data_dir, cluster_id);
        const fs::path summary_path = vector_db_v3::paths::final_cluster_summary_bin(data_dir, cluster_id);
        ok &= expect(fs::exists(manifest_path), "final cluster manifest exists");
        ok &= expect(fs::exists(assignments_path), "final cluster assignments exists");
        ok &= expect(fs::exists(summary_path), "final cluster summary exists");

        std::vector<vector_db_v3::codec::FinalAssignmentRow> final_rows;
        ok &= expect(vector_db_v3::codec::read_final_assignments_file(assignments_path, &final_rows).ok, "read final assignments");
        ok &= expect(vector_db_v3::codec::validate_final_assignments(final_rows).ok, "validate final assignments");
    }

    std::vector<vector_db_v3::codec::KSearchBoundsBatchRow> k_rows;
    ok &= expect(vector_db_v3::codec::read_k_search_bounds_batch_file(k_search_bin, &k_rows).ok, "read k_search_bounds_batch");
    ok &= expect(vector_db_v3::codec::validate_k_search_bounds_batch(k_rows).ok, "validate k_search_bounds_batch");
    bool saw_top = false;
    bool saw_mid = false;
    bool saw_lower = false;
    for (const auto& row : k_rows) {
        if (row.stage_level == vector_db_v3::codec::StageLevel::Top) {
            saw_top = true;
        } else if (row.stage_level == vector_db_v3::codec::StageLevel::Mid) {
            saw_mid = true;
        } else if (row.stage_level == vector_db_v3::codec::StageLevel::Lower) {
            saw_lower = true;
        }
    }
    ok &= expect(saw_top, "k_search rows include top");
    ok &= expect(saw_mid, "k_search rows include mid");
    ok &= expect(saw_lower, "k_search rows include lower");

    std::vector<vector_db_v3::codec::PostClusterMembershipRow> membership_rows;
    ok &= expect(
        vector_db_v3::codec::read_post_cluster_membership_file(post_membership_bin, &membership_rows).ok,
        "read post_cluster_membership");
    ok &= expect(
        vector_db_v3::codec::validate_post_cluster_membership(membership_rows, true).ok,
        "validate post_cluster_membership");
    ok &= expect(membership_rows.size() == 64U, "post_cluster_membership row-count");

    std::set<std::uint32_t> valid_final_ids = cluster_ids;
    valid_final_ids.insert(std::numeric_limits<std::uint32_t>::max());
    for (const auto& row : membership_rows) {
        ok &= expect(valid_final_ids.count(row.final_cluster_numeric_id) > 0U, "membership final id resolves");
    }

    ok &= expect(store.close().ok, "close");
    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_final_layer_artifacts_tests: PASS\n";
    return 0;
}

