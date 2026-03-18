#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <vector>

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
        v[i] = base + static_cast<float>((i % 3U) * 0.001f);
    }
    return v;
}

void set_env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

}  // namespace

int main() {
    const fs::path data_dir = fs::temp_directory_path() / "vectordb_v3_compliance_fail_test";
    std::error_code ec;
    fs::remove_all(data_dir, ec);

    bool ok = true;
    set_env("VECTOR_DB_V3_COMPLIANCE_PROFILE", "fail");
    vector_db_v3::VectorStore store(data_dir.string());
    ok &= expect(store.init().ok, "init");
    ok &= expect(store.open().ok, "open");
    ok &= expect(store.insert(1, make_vec(0.2f)).ok, "insert");

    ok &= expect(!store.build_top_clusters(8U).ok, "build_top_clusters fail-fast");
    ok &= expect(!store.build_mid_layer_clusters(8U).ok, "build_mid_layer_clusters fail-fast");
    ok &= expect(!store.build_lower_layer_clusters(8U).ok, "build_lower_layer_clusters fail-fast");
    ok &= expect(!store.build_final_layer_clusters(8U).ok, "build_final_layer_clusters fail-fast");

    const auto stats = store.cluster_stats();
    ok &= expect(stats.compliance_status == "fail", "compliance_status fail");
    ok &= expect(!stats.fallback_reason.empty(), "fallback_reason populated");
    ok &= expect(stats.fallback_reason == "profile_forced_fail", "fallback_reason profile_forced_fail");
    ok &= expect(!stats.non_compliance_stage.empty(), "non_compliance_stage populated");
    ok &= expect(store.close().ok, "close");
    set_env("VECTOR_DB_V3_COMPLIANCE_PROFILE", "");

    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_compliance_fail_fast_tests: PASS\n";
    return 0;
}

