#include <filesystem>
#include <iostream>
#include <unordered_map>
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
        v[i] = base + static_cast<float>((i % 11U) * 0.005f);
    }
    return v;
}

}  // namespace

int main() {
    const fs::path data_dir = fs::temp_directory_path() / "vectordb_v3_mid_layer_artifacts_test";
    std::error_code ec;
    fs::remove_all(data_dir, ec);

    bool ok = true;
    vector_db_v3::VectorStore store(data_dir.string());
    ok &= expect(store.init().ok, "init");
    ok &= expect(store.open().ok, "open");
    for (std::uint64_t id = 1; id <= 32; ++id) {
        ok &= expect(store.insert(id, make_vec(static_cast<float>(id) * 0.1f)).ok, "insert");
    }

    ok &= expect(store.build_top_clusters(123U).ok, "build_top_clusters");
    ok &= expect(store.build_mid_layer_clusters(123U).ok, "build_mid_layer_clusters");

    const fs::path top_assignments_path = vector_db_v3::paths::top_assignments_bin(data_dir);
    const fs::path mid_assignments_path = vector_db_v3::paths::mid_layer_assignments_bin(data_dir);
    const fs::path mid_summary_path = vector_db_v3::paths::mid_layer_clustering_bin(data_dir);

    ok &= expect(fs::exists(top_assignments_path), "top assignments exist");
    ok &= expect(fs::exists(mid_assignments_path), "mid assignments exist");
    ok &= expect(fs::exists(mid_summary_path), "mid summary exists");

    std::vector<vector_db_v3::codec::TopAssignmentRow> top_rows;
    ok &= expect(
        vector_db_v3::codec::read_top_assignments_file(top_assignments_path, &top_rows).ok,
        "read top assignments");

    std::vector<vector_db_v3::codec::MidAssignmentRow> mid_rows;
    ok &= expect(
        vector_db_v3::codec::read_mid_assignments_file(mid_assignments_path, &mid_rows).ok,
        "read mid assignments");
    ok &= expect(mid_rows.size() == top_rows.size(), "mid row-count equals top row-count");
    ok &= expect(mid_rows.size() == 32U, "mid row-count equals inserted rows");

    std::unordered_map<std::uint64_t, std::uint32_t> top_parent_by_id;
    top_parent_by_id.reserve(top_rows.size());
    for (const auto& row : top_rows) {
        top_parent_by_id[row.embedding_id] = row.top_centroid_numeric_id;
    }
    for (const auto& row : mid_rows) {
        const auto it = top_parent_by_id.find(row.embedding_id);
        ok &= expect(it != top_parent_by_id.end(), "mid embedding exists in top assignments");
        if (it != top_parent_by_id.end()) {
            ok &= expect(
                row.parent_top_centroid_numeric_id == it->second,
                "mid parent id matches top parent id");
        }
    }

    vector_db_v3::codec::CommonHeader summary_header{};
    std::vector<std::uint8_t> summary_payload;
    ok &= expect(
        vector_db_v3::codec::read_cluster_manifest_file(
            mid_summary_path,
            &summary_header,
            &summary_payload).ok,
        "read mid summary");
    ok &= expect(summary_header.schema_version == 1U, "summary schema version");
    const std::string payload_str(summary_payload.begin(), summary_payload.end());
    ok &= expect(payload_str.find("\"stage\":\"mid\"") != std::string::npos, "summary stage field");
    ok &= expect(
        payload_str.find("mid_layer_clustering/assignments.bin") != std::string::npos,
        "summary includes assignments artifact");
    ok &= expect(payload_str.find("\"parent_jobs\"") != std::string::npos, "summary includes parent_jobs");

    ok &= expect(store.close().ok, "close");
    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_mid_layer_artifacts_tests: PASS\n";
    return 0;
}
