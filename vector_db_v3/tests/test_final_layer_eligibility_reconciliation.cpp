#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
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
        v[i] = base + static_cast<float>((i % 7U) * 0.002f);
    }
    return v;
}

void set_lower_threshold_env(const char* value) {
#ifdef _WIN32
    _putenv_s("VECTOR_DB_V3_LOWER_GATE_THRESHOLD", value);
#else
    setenv("VECTOR_DB_V3_LOWER_GATE_THRESHOLD", value, 1);
#endif
}

}  // namespace

int main() {
    const fs::path data_dir = fs::temp_directory_path() / "vectordb_v3_final_layer_eligibility_test";
    std::error_code ec;
    fs::remove_all(data_dir, ec);

    bool ok = true;
    vector_db_v3::VectorStore store(data_dir.string());
    ok &= expect(store.init().ok, "init");
    ok &= expect(store.open().ok, "open");
    for (std::uint64_t id = 1; id <= 48; ++id) {
        ok &= expect(store.insert(id, make_vec(static_cast<float>(id) * 0.03f)).ok, "insert");
    }

    ok &= expect(store.build_top_clusters(13U).ok, "build_top_clusters");
    ok &= expect(store.build_mid_layer_clusters(13U).ok, "build_mid_layer_clusters");
    set_lower_threshold_env("0");
    ok &= expect(store.build_lower_layer_clusters(13U).ok, "build_lower_layer_clusters threshold=0");
    ok &= expect(store.build_final_layer_clusters(13U).ok, "build_final_layer_clusters");
    set_lower_threshold_env("");

    vector_db_v3::codec::CommonHeader final_hdr{};
    std::vector<std::uint8_t> final_payload;
    ok &= expect(
        vector_db_v3::codec::read_cluster_manifest_file(
            vector_db_v3::paths::final_layer_clusters_bin(data_dir), &final_hdr, &final_payload).ok,
        "read final aggregate");
    const std::string final_payload_s(final_payload.begin(), final_payload.end());
    ok &= expect(final_payload_s.find("\"eligible_branches_stop\":0") != std::string::npos, "no stop-eligible branches");

    std::vector<vector_db_v3::codec::KSearchBoundsBatchRow> k_rows;
    ok &= expect(
        vector_db_v3::codec::read_k_search_bounds_batch_file(
            vector_db_v3::paths::k_search_bounds_batch_bin(data_dir), &k_rows).ok,
        "read k_search_bounds_batch");
    bool saw_lower = false;
    for (const auto& row : k_rows) {
        if (row.stage_level == vector_db_v3::codec::StageLevel::Lower) {
            saw_lower = true;
            ok &= expect(
                row.gate_decision == vector_db_v3::codec::GateDecision::Continue,
                "lower rows are continue when threshold=0");
        }
    }
    ok &= expect(saw_lower, "lower rows exist in k_search_bounds_batch");

    std::vector<vector_db_v3::codec::PostClusterMembershipRow> membership_rows;
    ok &= expect(
        vector_db_v3::codec::read_post_cluster_membership_file(
            vector_db_v3::paths::post_cluster_membership_bin(data_dir), &membership_rows).ok,
        "read post_cluster_membership");
    ok &= expect(membership_rows.size() == 48U, "membership row-count");
    for (const auto& row : membership_rows) {
        ok &= expect(
            row.final_cluster_numeric_id == std::numeric_limits<std::uint32_t>::max(),
            "non-eligible rows use final sentinel");
    }

    ok &= expect(store.close().ok, "close");
    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_final_layer_eligibility_reconciliation_tests: PASS\n";
    return 0;
}

