#include <cstdlib>
#include <iostream>
#include <memory>
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
        v[i] = base + static_cast<float>((i % 13U) * 0.0007f);
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
    using namespace vector_db_v3;
    bool ok = true;

    std::vector<std::vector<float>> vectors;
    vectors.reserve(96U);
    for (std::size_t i = 0; i < 48U; ++i) {
        vectors.push_back(make_vec(0.15f));
    }
    for (std::size_t i = 0; i < 48U; ++i) {
        vectors.push_back(make_vec(0.85f));
    }

    set_env("VECTOR_DB_V3_GPU_RESIDENCY_MODE", "stage");
    set_env("VECTOR_DB_V3_GPU_RESIDENCY_MAX_BYTES", "268435456");

    auto ctx = kmeans::CudaPipelineContext::create_from_env();
    ok &= expect(ctx != nullptr, "create context");

    const bool cuda_ready = kmeans::cuda_backend_available(nullptr);
    if (cuda_ready && ctx != nullptr && ctx->enabled()) {
        kmeans::KMeansResult out1{};
        kmeans::KMeansResult out2{};
        std::string backend1;
        std::string backend2;
        Status st = kmeans::run_kmeans(
            vectors,
            8U,
            6U,
            kmeans::BackendPreference::Cuda,
            &out1,
            &backend1,
            ctx.get());
        ok &= expect(st.ok, "first residency cuda run should succeed");
        const auto s1 = ctx->stats();
        ok &= expect(s1.alloc_calls > 0U, "first run should allocate device buffers");

        st = kmeans::run_kmeans(
            vectors,
            8U,
            6U,
            kmeans::BackendPreference::Cuda,
            &out2,
            &backend2,
            ctx.get());
        ok &= expect(st.ok, "second residency cuda run should succeed");
        const auto s2 = ctx->stats();
        ok &= expect(s2.cache_hits >= s1.cache_hits, "cache_hits should be monotonic");
        ok &= expect(s2.cache_hits > 0U, "second run should show residency cache hits");
        ok &= expect(s2.alloc_calls <= s1.alloc_calls + 2U, "second run should avoid full realloc churn");
    }

    // Low-memory fallback path: auto backend should still succeed via CPU fallback.
    set_env("VECTOR_DB_V3_GPU_RESIDENCY_MODE", "stage");
    set_env("VECTOR_DB_V3_GPU_RESIDENCY_MAX_BYTES", "1024");
    auto tiny_ctx = kmeans::CudaPipelineContext::create_from_env();
    kmeans::KMeansResult tiny_out{};
    std::string tiny_backend;
    Status tiny_st = kmeans::run_kmeans(
        vectors,
        8U,
        4U,
        kmeans::BackendPreference::Auto,
        &tiny_out,
        &tiny_backend,
        tiny_ctx.get());
    ok &= expect(tiny_st.ok, "tiny-memory auto mode should remain successful");
    if (cuda_ready && tiny_ctx != nullptr && tiny_ctx->enabled()) {
        ok &= expect(tiny_backend == "cpu", "tiny-memory auto mode should fallback to cpu");
    }

    set_env("VECTOR_DB_V3_GPU_RESIDENCY_MODE", "");
    set_env("VECTOR_DB_V3_GPU_RESIDENCY_MAX_BYTES", "");

    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_gpu_residency_tests: PASS\n";
    return 0;
}
