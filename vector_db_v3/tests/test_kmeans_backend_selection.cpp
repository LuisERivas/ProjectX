#include <iostream>
#include <vector>

#include "vector_db_v3/kmeans_backend.hpp"

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
        v[i] = base + static_cast<float>((i % 11U) * 0.0005f);
    }
    return v;
}

}  // namespace

int main() {
    using namespace vector_db_v3;
    bool ok = true;

    std::vector<std::vector<float>> vectors;
    for (std::size_t i = 0; i < 24U; ++i) {
        vectors.push_back(make_vec(0.2f));
    }
    for (std::size_t i = 0; i < 24U; ++i) {
        vectors.push_back(make_vec(0.8f));
    }

    kmeans::KMeansResult out{};
    std::string backend;

    Status st = kmeans::run_kmeans(vectors, 6U, 6U, kmeans::BackendPreference::Auto, &out, &backend);
    ok &= expect(st.ok, "auto backend should succeed");
    const bool cuda_ready = kmeans::cuda_backend_available(nullptr);
    if (cuda_ready) {
        ok &= expect(
            backend == "cuda_fp32" || backend == "cuda_tensor_fp16",
            "auto backend should select a cuda backend path when available");
    } else {
        ok &= expect(backend == "cpu", "auto backend should fallback to cpu when cuda unavailable");
    }

    st = kmeans::run_kmeans(vectors, 6U, 6U, kmeans::BackendPreference::Cpu, &out, &backend);
    ok &= expect(st.ok, "cpu preference should always succeed");
    ok &= expect(backend == "cpu", "cpu preference should report cpu backend");

    st = kmeans::run_kmeans(vectors, 6U, 6U, kmeans::BackendPreference::Cuda, &out, &backend);
    if (cuda_ready) {
        ok &= expect(st.ok, "cuda preference should succeed when runtime is ready");
        ok &= expect(
            backend == "cuda_fp32" || backend == "cuda_tensor_fp16",
            "cuda preference should report a concrete cuda backend path");
    } else {
        ok &= expect(!st.ok, "cuda preference should fail when cuda is unavailable");
    }

    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_kmeans_backend_selection_tests: PASS\n";
    return 0;
}
