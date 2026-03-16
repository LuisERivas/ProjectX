#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

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
        v[i] = base + static_cast<float>((i % 9U) * 0.003f);
    }
    return v;
}

std::size_t count_occurrences(const std::string& haystack, const std::string& needle) {
    std::size_t count = 0;
    std::size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
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
    const fs::path data_dir = fs::temp_directory_path() / "vectordb_v3_lower_layer_artifacts_test";
    std::error_code ec;
    fs::remove_all(data_dir, ec);

    bool ok = true;
    vector_db_v3::VectorStore store(data_dir.string());
    ok &= expect(store.init().ok, "init");
    ok &= expect(store.open().ok, "open");
    for (std::uint64_t id = 1; id <= 64; ++id) {
        ok &= expect(store.insert(id, make_vec(static_cast<float>(id) * 0.07f)).ok, "insert");
    }

    ok &= expect(store.build_top_clusters(11U).ok, "build_top_clusters");
    ok &= expect(store.build_mid_layer_clusters(11U).ok, "build_mid_layer_clusters");
    ok &= expect(store.build_lower_layer_clusters(11U).ok, "build_lower_layer_clusters");

    const fs::path lower_summary = vector_db_v3::paths::lower_layer_clustering_bin(data_dir);
    ok &= expect(fs::exists(lower_summary), "LOWER_LAYER_CLUSTERING.bin exists");

    vector_db_v3::codec::CommonHeader hdr{};
    std::vector<std::uint8_t> payload;
    ok &= expect(
        vector_db_v3::codec::read_cluster_manifest_file(lower_summary, &hdr, &payload).ok,
        "read lower summary");
    ok &= expect(hdr.schema_version == 1U, "lower summary schema version");

    const std::string payload_str(payload.begin(), payload.end());
    ok &= expect(payload_str.find("\"stage\":\"lower\"") != std::string::npos, "lower stage tag");
    ok &= expect(payload_str.find("\"gate_threshold\":") != std::string::npos, "gate threshold present");
    ok &= expect(payload_str.find("\"gate_outcomes\"") != std::string::npos, "gate outcomes present");
    ok &= expect(payload_str.find("\"gate_decision\":\"stop\"") != std::string::npos, "stop outcomes present");

    const std::size_t stop_count = count_occurrences(payload_str, "\"gate_decision\":\"stop\"");
    ok &= expect(stop_count > 0U, "stop_count positive");

    set_lower_threshold_env("0");
    ok &= expect(store.build_lower_layer_clusters(11U).ok, "build_lower_layer_clusters threshold=0");
    vector_db_v3::codec::CommonHeader hdr2{};
    std::vector<std::uint8_t> payload2;
    ok &= expect(
        vector_db_v3::codec::read_cluster_manifest_file(lower_summary, &hdr2, &payload2).ok,
        "read lower summary threshold=0");
    const std::string payload_str2(payload2.begin(), payload2.end());
    ok &= expect(payload_str2.find("\"gate_decision\":\"continue\"") != std::string::npos, "continue outcomes present");
    const std::size_t continue_count = count_occurrences(payload_str2, "\"gate_decision\":\"continue\"");
    ok &= expect(continue_count > 0U, "continue_count positive");
    set_lower_threshold_env("");

    ok &= expect(store.close().ok, "close");
    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_lower_layer_gate_artifacts_tests: PASS\n";
    return 0;
}
