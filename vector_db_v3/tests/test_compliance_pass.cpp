#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
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
        v[i] = base + static_cast<float>((i % 5U) * 0.002f);
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
    const fs::path data_dir = fs::temp_directory_path() / "vectordb_v3_compliance_pass_test";
    std::error_code ec;
    fs::remove_all(data_dir, ec);

    bool ok = true;
    set_env("VECTOR_DB_V3_COMPLIANCE_PROFILE", "pass");
    vector_db_v3::VectorStore store(data_dir.string());
    ok &= expect(store.init().ok, "init");
    ok &= expect(store.open().ok, "open");
    ok &= expect(store.insert(1, make_vec(0.1f)).ok, "insert");
    ok &= expect(store.build_top_clusters(7U).ok, "build_top_clusters");

    const auto stats = store.cluster_stats();
    ok &= expect(stats.compliance_status == "pass", "compliance_status pass");
    ok &= expect(stats.cuda_required, "cuda_required true");
    ok &= expect(stats.cuda_enabled, "cuda_enabled true");
    ok &= expect(stats.gpu_arch_class == "ampere", "gpu_arch_class ampere");
    ok &= expect(
        stats.kernel_backend_path == "cuda_tensor_fp16" || stats.kernel_backend_path == "cuda_fp32",
        "kernel_backend_path should report concrete cuda backend");
    if (stats.kernel_backend_path == "cuda_tensor_fp16") {
        ok &= expect(stats.tensor_core_active, "tensor_core_active true on tensor backend");
    } else {
        ok &= expect(!stats.tensor_core_active, "tensor_core_active false on fp32 backend");
    }
    ok &= expect(stats.hot_path_language == "cpp_cuda", "hot_path_language cpp_cuda");
    ok &= expect(store.close().ok, "close");
    set_env("VECTOR_DB_V3_COMPLIANCE_PROFILE", "");

    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_compliance_pass_tests: PASS\n";
    return 0;
}

