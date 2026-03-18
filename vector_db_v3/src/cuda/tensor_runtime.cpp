#include "tensor_runtime.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#if VECTOR_DB_V3_CUDA_ENABLED
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "kmeans_kernels.hpp"
#include "tensor_distance.hpp"
#endif

namespace vector_db_v3::kmeans {

namespace {

bool env_truthy(const char* value) {
    if (value == nullptr) {
        return false;
    }
    std::string s(value);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

#if VECTOR_DB_V3_CUDA_ENABLED
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    Status alloc(std::size_t bytes) {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        if (bytes == 0U) {
            return Status::Ok();
        }
        const cudaError_t st = cudaMalloc(&ptr_, bytes);
        if (st != cudaSuccess) {
            ptr_ = nullptr;
            return Status::Error(std::string("cuda malloc failed: ") + cudaGetErrorString(st));
        }
        return Status::Ok();
    }

    void* get() const {
        return ptr_;
    }

private:
    void* ptr_ = nullptr;
};

Status checked_copy_h2d(void* dst, const void* src, std::size_t bytes, const char* label) {
    if (bytes == 0U) {
        return Status::Ok();
    }
    const cudaError_t st = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) {
        return Status::Error(std::string("cuda memcpy h2d failed (") + label + "): " + cudaGetErrorString(st));
    }
    return Status::Ok();
}

Status checked_copy_d2h(void* dst, const void* src, std::size_t bytes, const char* label) {
    if (bytes == 0U) {
        return Status::Ok();
    }
    const cudaError_t st = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) {
        return Status::Error(std::string("cuda memcpy d2h failed (") + label + "): " + cudaGetErrorString(st));
    }
    return Status::Ok();
}

Status checked_memset(void* ptr, int value, std::size_t bytes, const char* label) {
    if (bytes == 0U) {
        return Status::Ok();
    }
    const cudaError_t st = cudaMemset(ptr, value, bytes);
    if (st != cudaSuccess) {
        return Status::Error(std::string("cuda memset failed (") + label + "): " + cudaGetErrorString(st));
    }
    return Status::Ok();
}

Status checked_sync(const char* label) {
    const cudaError_t st = cudaDeviceSynchronize();
    if (st != cudaSuccess) {
        return Status::Error(std::string("cuda sync failed (") + label + "): " + cudaGetErrorString(st));
    }
    return Status::Ok();
}

#endif

}  // namespace

bool tensor_path_effective(std::uint32_t vector_count, std::uint32_t k, std::size_t dim, std::string* reason) {
    // Tensor path is not effective for tiny clustering problems and can add
    // avoidable launch/setup overhead relative to FP32 CUDA.
    if (vector_count < 8U || k < 2U) {
        if (reason != nullptr) {
            *reason = "tensor_problem_too_small";
        }
        return false;
    }
    const char* override = std::getenv("VECTOR_DB_V3_FORCE_TENSOR_PATH");
    if (env_truthy(override)) {
        if (reason != nullptr) {
            *reason = "forced_tensor_path";
        }
        return true;
    }
    std::uint64_t min_ops = 8ULL * 1024ULL * 1024ULL;
    if (const char* env_min_ops = std::getenv("VECTOR_DB_V3_TENSOR_MIN_OPS")) {
        try {
            min_ops = static_cast<std::uint64_t>(std::stoull(env_min_ops));
        } catch (...) {
            min_ops = 8ULL * 1024ULL * 1024ULL;
        }
    }
    const std::uint64_t ops = static_cast<std::uint64_t>(vector_count) * static_cast<std::uint64_t>(k) *
        static_cast<std::uint64_t>(dim);
    const bool effective = ops >= min_ops;
    if (reason != nullptr) {
        *reason = effective ? "tensor_effective" : "tensor_path_not_effective";
    }
    return effective;
}

Status run_kmeans_cuda_tensor(
    const std::vector<std::vector<float>>& vectors,
    std::uint32_t k,
    std::uint32_t max_iterations,
    CudaPipelineContext* pipeline_context,
    KMeansResult* out) {
#if VECTOR_DB_V3_CUDA_ENABLED
#if VECTOR_DB_V3_TENSOR_ENABLED
    if (out == nullptr) {
        return Status::Error("kmeans tensor: out is null");
    }
    *out = KMeansResult{};
    if (vectors.empty() || k == 0U) {
        return Status::Ok();
    }
    for (const auto& vec : vectors) {
        if (vec.size() != kVectorDim) {
            return Status::Error("kmeans tensor: non-1024D vector encountered");
        }
    }

    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        return Status::Error("kmeans tensor: cublasCreate failed");
    }

    const std::uint32_t n = static_cast<std::uint32_t>(vectors.size());
    k = std::min<std::uint32_t>(k, n);
    const std::size_t dim = kVectorDim;

    std::vector<float> h_vectors(static_cast<std::size_t>(n) * dim, 0.0f);
    for (std::uint32_t i = 0; i < n; ++i) {
        std::copy(vectors[i].begin(), vectors[i].end(), h_vectors.begin() + static_cast<std::size_t>(i) * dim);
    }
    std::vector<float> h_centroids(static_cast<std::size_t>(k) * dim, 0.0f);
    for (std::uint32_t c = 0; c < k; ++c) {
        const std::size_t idx = (static_cast<std::size_t>(c) * vectors.size()) / k;
        std::copy(vectors[idx].begin(), vectors[idx].end(), h_centroids.begin() + static_cast<std::size_t>(c) * dim);
    }

    std::vector<std::uint32_t> h_assignments(n, 0U);
    std::vector<float> h_min_dists(n, std::numeric_limits<float>::infinity());
    std::vector<std::uint32_t> h_counts(k, 0U);
    std::vector<std::uint32_t> prev_assignments(n, std::numeric_limits<std::uint32_t>::max());

    DeviceBuffer d_vectors_f32;
    DeviceBuffer d_centroids_f32;
    DeviceBuffer d_vectors_fp16_col;
    DeviceBuffer d_centroids_fp16_col;
    DeviceBuffer d_dot_kn;
    DeviceBuffer d_vector_norms;
    DeviceBuffer d_centroid_norms;
    DeviceBuffer d_assignments;
    DeviceBuffer d_min_dists;
    DeviceBuffer d_sums;
    DeviceBuffer d_counts;

    void* d_vectors_f32_ptr = nullptr;
    void* d_centroids_f32_ptr = nullptr;
    void* d_vectors_fp16_col_ptr = nullptr;
    void* d_centroids_fp16_col_ptr = nullptr;
    void* d_dot_kn_ptr = nullptr;
    void* d_vector_norms_ptr = nullptr;
    void* d_centroid_norms_ptr = nullptr;
    void* d_assignments_ptr = nullptr;
    void* d_min_dists_ptr = nullptr;
    void* d_sums_ptr = nullptr;
    void* d_counts_ptr = nullptr;

    const bool residency_enabled = pipeline_context != nullptr && pipeline_context->enabled();
    const std::uint64_t vectors_token =
        vectors.empty() ? 0ULL : (reinterpret_cast<std::uint64_t>(vectors.data()) ^
                                  (reinterpret_cast<std::uint64_t>(vectors.front().data()) << 1U) ^
                                  (static_cast<std::uint64_t>(vectors.size()) << 32U));

    auto acquire = [&](const char* key, std::size_t bytes, DeviceBuffer* local, void** out_ptr) -> Status {
        if (residency_enabled) {
            bool reused = false;
            return pipeline_context->acquire_buffer(key, bytes, out_ptr, &reused);
        }
        Status st_local = local->alloc(bytes);
        if (!st_local.ok) {
            return st_local;
        }
        *out_ptr = local->get();
        return Status::Ok();
    };
    auto copy_h2d = [&](void* dst, const void* src, std::size_t bytes, const char* label) -> Status {
        if (residency_enabled) {
            return pipeline_context->copy_h2d(dst, src, bytes, label);
        }
        return checked_copy_h2d(dst, src, bytes, label);
    };
    auto copy_d2h = [&](void* dst, const void* src, std::size_t bytes, const char* label) -> Status {
        if (residency_enabled) {
            return pipeline_context->copy_d2h(dst, src, bytes, label);
        }
        return checked_copy_d2h(dst, src, bytes, label);
    };
    auto memset_dev = [&](void* ptr, int value, std::size_t bytes, const char* label) -> Status {
        if (residency_enabled) {
            return pipeline_context->memset(ptr, value, bytes, label);
        }
        return checked_memset(ptr, value, bytes, label);
    };
    auto sync_dev = [&](const char* label) -> Status {
        if (residency_enabled) {
            return pipeline_context->sync(label);
        }
        return checked_sync(label);
    };

    Status st = acquire("tensor_vectors_f32", h_vectors.size() * sizeof(float), &d_vectors_f32, &d_vectors_f32_ptr);
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }
    st = acquire("tensor_centroids_f32", h_centroids.size() * sizeof(float), &d_centroids_f32, &d_centroids_f32_ptr);
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }
    st = acquire(
        "tensor_vectors_fp16_col",
        h_vectors.size() * sizeof(std::uint16_t),
        &d_vectors_fp16_col,
        &d_vectors_fp16_col_ptr);
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }
    st = acquire(
        "tensor_centroids_fp16_col",
        h_centroids.size() * sizeof(std::uint16_t),
        &d_centroids_fp16_col,
        &d_centroids_fp16_col_ptr);
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }
    st = acquire("tensor_dot_kn", static_cast<std::size_t>(k) * n * sizeof(float), &d_dot_kn, &d_dot_kn_ptr);
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }
    st = acquire("tensor_vector_norms", n * sizeof(float), &d_vector_norms, &d_vector_norms_ptr);
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }
    st = acquire("tensor_centroid_norms", k * sizeof(float), &d_centroid_norms, &d_centroid_norms_ptr);
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }
    st = acquire("tensor_assignments", n * sizeof(std::uint32_t), &d_assignments, &d_assignments_ptr);
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }
    st = acquire("tensor_min_dists", n * sizeof(float), &d_min_dists, &d_min_dists_ptr);
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }
    st = acquire("tensor_sums", h_centroids.size() * sizeof(float), &d_sums, &d_sums_ptr);
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }
    st = acquire("tensor_counts", k * sizeof(std::uint32_t), &d_counts, &d_counts_ptr);
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }

    bool vectors_h2d_skipped = false;
    if (residency_enabled) {
        st = pipeline_context->copy_h2d_if_changed(
            "tensor_vectors_f32",
            vectors_token,
            d_vectors_f32_ptr,
            h_vectors.data(),
            h_vectors.size() * sizeof(float),
            "vectors",
            &vectors_h2d_skipped);
    } else {
        st = copy_h2d(d_vectors_f32_ptr, h_vectors.data(), h_vectors.size() * sizeof(float), "vectors");
    }
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }
    st = copy_h2d(d_centroids_f32_ptr, h_centroids.data(), h_centroids.size() * sizeof(float), "centroids");
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }

    if (!vectors_h2d_skipped) {
        if (!cuda::launch_pack_rowmajor_to_colmajor_half(
                static_cast<const float*>(d_vectors_f32_ptr),
                static_cast<__half*>(d_vectors_fp16_col_ptr),
                n,
                dim)) {
            cublasDestroy(handle);
            return Status::Error("kmeans tensor: vectors fp16 pack launch failed");
        }
        st = sync_dev("pack_vectors");
        if (!st.ok) {
            cublasDestroy(handle);
            return st;
        }
    }
    if (!cuda::launch_row_norms_f32(
            static_cast<const float*>(d_vectors_f32_ptr),
            static_cast<float*>(d_vector_norms_ptr),
            n,
            dim)) {
        cublasDestroy(handle);
        return Status::Error("kmeans tensor: vector norms launch failed");
    }
    st = sync_dev("vector_norms");
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const std::uint32_t iters = std::max<std::uint32_t>(1U, max_iterations);
    for (std::uint32_t iter = 0; iter < iters; ++iter) {
        if (!cuda::launch_pack_rowmajor_to_colmajor_half(
                static_cast<const float*>(d_centroids_f32_ptr),
                static_cast<__half*>(d_centroids_fp16_col_ptr),
                k,
                dim)) {
            cublasDestroy(handle);
            return Status::Error("kmeans tensor: centroids fp16 pack launch failed");
        }
        st = sync_dev("pack_centroids");
        if (!st.ok) {
            cublasDestroy(handle);
            return st;
        }
        if (!cuda::launch_row_norms_f32(
                static_cast<const float*>(d_centroids_f32_ptr),
                static_cast<float*>(d_centroid_norms_ptr),
                k,
                dim)) {
            cublasDestroy(handle);
            return Status::Error("kmeans tensor: centroid norms launch failed");
        }
        st = sync_dev("centroid_norms");
        if (!st.ok) {
            cublasDestroy(handle);
            return st;
        }

        const cublasStatus_t gemm_st = cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(k),
            static_cast<int>(n),
            static_cast<int>(dim),
            &alpha,
            d_centroids_fp16_col_ptr,
            CUDA_R_16F,
            static_cast<int>(dim),
            d_vectors_fp16_col_ptr,
            CUDA_R_16F,
            static_cast<int>(dim),
            &beta,
            d_dot_kn_ptr,
            CUDA_R_32F,
            static_cast<int>(k),
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT);
        if (gemm_st != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handle);
            return Status::Error("kmeans tensor: cublasGemmEx failed");
        }
        st = sync_dev("gemm_dot");
        if (!st.ok) {
            cublasDestroy(handle);
            return st;
        }

        if (!cuda::launch_argmin_from_dot_kn(
                static_cast<const float*>(d_dot_kn_ptr),
                static_cast<const float*>(d_vector_norms_ptr),
                static_cast<const float*>(d_centroid_norms_ptr),
                static_cast<std::uint32_t*>(d_assignments_ptr),
                static_cast<float*>(d_min_dists_ptr),
                n,
                k)) {
            cublasDestroy(handle);
            return Status::Error("kmeans tensor: argmin launch failed");
        }
        st = sync_dev("argmin");
        if (!st.ok) {
            cublasDestroy(handle);
            return st;
        }

        st = copy_d2h(h_assignments.data(), d_assignments_ptr, n * sizeof(std::uint32_t), "assignments");
        if (!st.ok) {
            cublasDestroy(handle);
            return st;
        }
        st = copy_d2h(h_min_dists.data(), d_min_dists_ptr, n * sizeof(float), "min_dists");
        if (!st.ok) {
            cublasDestroy(handle);
            return st;
        }

        bool changed = (iter == 0U);
        if (!changed) {
            for (std::uint32_t i = 0; i < n; ++i) {
                if (h_assignments[i] != prev_assignments[i]) {
                    changed = true;
                    break;
                }
            }
        }
        prev_assignments = h_assignments;

        st = memset_dev(d_sums_ptr, 0, h_centroids.size() * sizeof(float), "sums");
        if (!st.ok) {
            cublasDestroy(handle);
            return st;
        }
        st = memset_dev(d_counts_ptr, 0, k * sizeof(std::uint32_t), "counts");
        if (!st.ok) {
            cublasDestroy(handle);
            return st;
        }
        if (!cuda::launch_accumulate_kernel(
                static_cast<const float*>(d_vectors_f32_ptr),
                static_cast<const std::uint32_t*>(d_assignments_ptr),
                static_cast<float*>(d_sums_ptr),
                static_cast<std::uint32_t*>(d_counts_ptr),
                n,
                k,
                dim)) {
            cublasDestroy(handle);
            return Status::Error("kmeans tensor: accumulate launch failed");
        }
        st = sync_dev("accumulate");
        if (!st.ok) {
            cublasDestroy(handle);
            return st;
        }

        st = copy_d2h(h_counts.data(), d_counts_ptr, k * sizeof(std::uint32_t), "counts");
        if (!st.ok) {
            cublasDestroy(handle);
            return st;
        }

        bool repaired = false;
        for (std::uint32_t c = 0; c < k; ++c) {
            if (h_counts[c] != 0U) {
                continue;
            }
            std::size_t worst_idx = 0U;
            float worst_dist = -1.0f;
            for (std::size_t i = 0; i < h_min_dists.size(); ++i) {
                if (h_min_dists[i] > worst_dist) {
                    worst_dist = h_min_dists[i];
                    worst_idx = i;
                }
            }
            h_assignments[worst_idx] = c;
            repaired = true;
            changed = true;
        }
        if (repaired) {
            st = copy_h2d(d_assignments_ptr, h_assignments.data(), n * sizeof(std::uint32_t), "repaired_assignments");
            if (!st.ok) {
                cublasDestroy(handle);
                return st;
            }
            st = memset_dev(d_sums_ptr, 0, h_centroids.size() * sizeof(float), "sums_repair");
            if (!st.ok) {
                cublasDestroy(handle);
                return st;
            }
            st = memset_dev(d_counts_ptr, 0, k * sizeof(std::uint32_t), "counts_repair");
            if (!st.ok) {
                cublasDestroy(handle);
                return st;
            }
            if (!cuda::launch_accumulate_kernel(
                    static_cast<const float*>(d_vectors_f32_ptr),
                    static_cast<const std::uint32_t*>(d_assignments_ptr),
                    static_cast<float*>(d_sums_ptr),
                    static_cast<std::uint32_t*>(d_counts_ptr),
                    n,
                    k,
                    dim)) {
                cublasDestroy(handle);
                return Status::Error("kmeans tensor: accumulate relaunch failed");
            }
            st = sync_dev("accumulate_repair");
            if (!st.ok) {
                cublasDestroy(handle);
                return st;
            }
        }

        if (!cuda::launch_update_kernel(
                static_cast<float*>(d_centroids_f32_ptr),
                static_cast<const float*>(d_sums_ptr),
                static_cast<const std::uint32_t*>(d_counts_ptr),
                k,
                dim)) {
            cublasDestroy(handle);
            return Status::Error("kmeans tensor: update launch failed");
        }
        st = sync_dev("update");
        if (!st.ok) {
            cublasDestroy(handle);
            return st;
        }

        if (!changed) {
            break;
        }
    }

    st = copy_d2h(h_centroids.data(), d_centroids_f32_ptr, h_centroids.size() * sizeof(float), "centroids_out");
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }
    st = copy_d2h(h_assignments.data(), d_assignments_ptr, n * sizeof(std::uint32_t), "assignments_out");
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }
    st = copy_d2h(h_min_dists.data(), d_min_dists_ptr, n * sizeof(float), "min_dists_out");
    if (!st.ok) {
        cublasDestroy(handle);
        return st;
    }

    out->centroids.assign(k, std::vector<float>(dim, 0.0f));
    for (std::uint32_t c = 0; c < k; ++c) {
        std::copy(
            h_centroids.begin() + static_cast<std::size_t>(c) * dim,
            h_centroids.begin() + static_cast<std::size_t>(c + 1U) * dim,
            out->centroids[c].begin());
    }
    out->assignments = std::move(h_assignments);
    out->objective = 0.0;
    for (float d : h_min_dists) {
        out->objective += static_cast<double>(d);
    }

    cublasDestroy(handle);
    return Status::Ok();
#else
    (void)vectors;
    (void)k;
    (void)max_iterations;
    (void)pipeline_context;
    (void)out;
    return Status::Error("kmeans tensor unavailable: tensor_backend_not_compiled");
#endif
#else
    (void)vectors;
    (void)k;
    (void)max_iterations;
    (void)pipeline_context;
    (void)out;
    return Status::Error("kmeans tensor unavailable: cuda_not_compiled");
#endif
}

}  // namespace vector_db_v3::kmeans
