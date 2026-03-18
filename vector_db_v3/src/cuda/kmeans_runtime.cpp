#include "vector_db_v3/kmeans_backend.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#if VECTOR_DB_V3_CUDA_ENABLED
#include <cuda_runtime_api.h>
#include "kmeans_kernels.hpp"
#include "tensor_runtime.hpp"
#endif

namespace vector_db_v3::kmeans {

namespace {

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

std::uint64_t host_token_for_vectors(const std::vector<std::vector<float>>& vectors) {
    const std::uint64_t n = static_cast<std::uint64_t>(vectors.size());
    if (vectors.empty()) {
        return (n << 32U) ^ 0x9e3779b97f4a7c15ULL;
    }
    const std::uint64_t p0 = reinterpret_cast<std::uint64_t>(vectors.data());
    const std::uint64_t p1 = reinterpret_cast<std::uint64_t>(vectors.front().data());
    return p0 ^ (p1 << 1U) ^ (n << 32U);
}

Status run_kmeans_cuda_impl(
    const std::vector<std::vector<float>>& vectors,
    std::uint32_t k,
    std::uint32_t max_iterations,
    CudaPipelineContext* pipeline_context,
    KMeansResult* out) {
    if (out == nullptr) {
        return Status::Error("kmeans cuda: out is null");
    }
    *out = KMeansResult{};
    if (vectors.empty() || k == 0U) {
        return Status::Ok();
    }
    for (const auto& vec : vectors) {
        if (vec.size() != kVectorDim) {
            return Status::Error("kmeans cuda: non-1024D vector encountered");
        }
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
    std::vector<float> h_objective_terms(n, 0.0f);
    std::vector<std::uint32_t> prev_assignments(n, std::numeric_limits<std::uint32_t>::max());

    DeviceBuffer d_vectors;
    DeviceBuffer d_centroids;
    DeviceBuffer d_assignments;
    DeviceBuffer d_min_dists;
    DeviceBuffer d_sums;
    DeviceBuffer d_counts;
    DeviceBuffer d_objective_terms;

    void* d_vectors_ptr = nullptr;
    void* d_centroids_ptr = nullptr;
    void* d_assignments_ptr = nullptr;
    void* d_min_dists_ptr = nullptr;
    void* d_sums_ptr = nullptr;
    void* d_counts_ptr = nullptr;
    void* d_objective_terms_ptr = nullptr;
    const bool residency_enabled = pipeline_context != nullptr && pipeline_context->enabled();

    auto acquire = [&](const char* key, std::size_t bytes, DeviceBuffer* local, void** out_ptr) -> Status {
        if (residency_enabled) {
            bool reused = false;
            return pipeline_context->acquire_buffer(key, bytes, out_ptr, &reused);
        }
        Status st = local->alloc(bytes);
        if (!st.ok) {
            return st;
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

    Status st = acquire("kmeans_vectors_f32", h_vectors.size() * sizeof(float), &d_vectors, &d_vectors_ptr);
    if (!st.ok) {
        return st;
    }
    st = acquire("kmeans_centroids_f32", h_centroids.size() * sizeof(float), &d_centroids, &d_centroids_ptr);
    if (!st.ok) {
        return st;
    }
    st = acquire("kmeans_assignments", h_assignments.size() * sizeof(std::uint32_t), &d_assignments, &d_assignments_ptr);
    if (!st.ok) {
        return st;
    }
    st = acquire("kmeans_min_dists", h_min_dists.size() * sizeof(float), &d_min_dists, &d_min_dists_ptr);
    if (!st.ok) {
        return st;
    }
    st = acquire("kmeans_sums", h_centroids.size() * sizeof(float), &d_sums, &d_sums_ptr);
    if (!st.ok) {
        return st;
    }
    st = acquire("kmeans_counts", h_counts.size() * sizeof(std::uint32_t), &d_counts, &d_counts_ptr);
    if (!st.ok) {
        return st;
    }
    st = acquire(
        "kmeans_objective_terms",
        h_objective_terms.size() * sizeof(float),
        &d_objective_terms,
        &d_objective_terms_ptr);
    if (!st.ok) {
        return st;
    }

    if (residency_enabled) {
        bool skipped = false;
        st = pipeline_context->copy_h2d_if_changed(
            "kmeans_vectors_f32",
            host_token_for_vectors(vectors),
            d_vectors_ptr,
            h_vectors.data(),
            h_vectors.size() * sizeof(float),
            "vectors",
            &skipped);
    } else {
        st = copy_h2d(d_vectors_ptr, h_vectors.data(), h_vectors.size() * sizeof(float), "vectors");
    }
    if (!st.ok) {
        return st;
    }
    st = copy_h2d(d_centroids_ptr, h_centroids.data(), h_centroids.size() * sizeof(float), "centroids");
    if (!st.ok) {
        return st;
    }

    const std::uint32_t iters = std::max<std::uint32_t>(1U, max_iterations);
    for (std::uint32_t iter = 0; iter < iters; ++iter) {
        if (!cuda::launch_assignment_kernel(
                static_cast<const float*>(d_vectors_ptr),
                static_cast<const float*>(d_centroids_ptr),
                static_cast<std::uint32_t*>(d_assignments_ptr),
                static_cast<float*>(d_min_dists_ptr),
                n,
                k,
                dim)) {
            return Status::Error("kmeans cuda: assignment kernel launch failed");
        }
        st = copy_d2h(h_assignments.data(), d_assignments_ptr, h_assignments.size() * sizeof(std::uint32_t), "assignments");
        if (!st.ok) {
            return st;
        }
        st = copy_d2h(h_min_dists.data(), d_min_dists_ptr, h_min_dists.size() * sizeof(float), "min_dists");
        if (!st.ok) {
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
            return st;
        }
        st = memset_dev(d_counts_ptr, 0, h_counts.size() * sizeof(std::uint32_t), "counts");
        if (!st.ok) {
            return st;
        }
        if (!cuda::launch_accumulate_kernel(
                static_cast<const float*>(d_vectors_ptr),
                static_cast<const std::uint32_t*>(d_assignments_ptr),
                static_cast<float*>(d_sums_ptr),
                static_cast<std::uint32_t*>(d_counts_ptr),
                n,
                k,
                dim)) {
            return Status::Error("kmeans cuda: accumulate kernel launch failed");
        }

        st = copy_d2h(h_counts.data(), d_counts_ptr, h_counts.size() * sizeof(std::uint32_t), "counts");
        if (!st.ok) {
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
            st = copy_h2d(
                d_assignments_ptr,
                h_assignments.data(),
                h_assignments.size() * sizeof(std::uint32_t),
                "repaired_assignments");
            if (!st.ok) {
                return st;
            }
            st = memset_dev(d_sums_ptr, 0, h_centroids.size() * sizeof(float), "sums_repair");
            if (!st.ok) {
                return st;
            }
            st = memset_dev(d_counts_ptr, 0, h_counts.size() * sizeof(std::uint32_t), "counts_repair");
            if (!st.ok) {
                return st;
            }
            if (!cuda::launch_accumulate_kernel(
                    static_cast<const float*>(d_vectors_ptr),
                    static_cast<const std::uint32_t*>(d_assignments_ptr),
                    static_cast<float*>(d_sums_ptr),
                    static_cast<std::uint32_t*>(d_counts_ptr),
                    n,
                    k,
                    dim)) {
                return Status::Error("kmeans cuda: accumulate kernel relaunch failed");
            }
        }

        if (!cuda::launch_update_kernel(
                static_cast<float*>(d_centroids_ptr),
                static_cast<const float*>(d_sums_ptr),
                static_cast<const std::uint32_t*>(d_counts_ptr),
                k,
                dim)) {
            return Status::Error("kmeans cuda: update kernel launch failed");
        }

        if (!changed) {
            break;
        }
    }

    if (!cuda::launch_objective_kernel(
            static_cast<const float*>(d_vectors_ptr),
            static_cast<const float*>(d_centroids_ptr),
            static_cast<const std::uint32_t*>(d_assignments_ptr),
            static_cast<float*>(d_objective_terms_ptr),
            n,
            dim)) {
        return Status::Error("kmeans cuda: objective kernel launch failed");
    }

    st = copy_d2h(h_centroids.data(), d_centroids_ptr, h_centroids.size() * sizeof(float), "centroids_out");
    if (!st.ok) {
        return st;
    }
    st = copy_d2h(h_assignments.data(), d_assignments_ptr, h_assignments.size() * sizeof(std::uint32_t), "assignments_out");
    if (!st.ok) {
        return st;
    }
    st = copy_d2h(
        h_objective_terms.data(),
        d_objective_terms_ptr,
        h_objective_terms.size() * sizeof(float),
        "objective_out");
    if (!st.ok) {
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
    for (float term : h_objective_terms) {
        out->objective += static_cast<double>(term);
    }
    return Status::Ok();
}
#endif

}  // namespace

bool cuda_backend_compiled() {
#if VECTOR_DB_V3_CUDA_ENABLED
    return true;
#else
    return false;
#endif
}

bool cuda_backend_available(std::string* reason) {
#if VECTOR_DB_V3_CUDA_ENABLED
    int device_count = 0;
    const cudaError_t count_st = cudaGetDeviceCount(&device_count);
    if (count_st != cudaSuccess || device_count <= 0) {
        if (reason != nullptr) {
            *reason = "no_cuda_device";
        }
        return false;
    }
    cudaDeviceProp prop{};
    const cudaError_t prop_st = cudaGetDeviceProperties(&prop, 0);
    if (prop_st != cudaSuccess) {
        if (reason != nullptr) {
            *reason = "device_properties_unavailable";
        }
        return false;
    }
    if (prop.major < 8) {
        if (reason != nullptr) {
            *reason = "gpu_arch_not_ampere";
        }
        return false;
    }
    if (reason != nullptr) {
        *reason = "ok";
    }
    return true;
#else
    if (reason != nullptr) {
        *reason = "cuda_not_compiled";
    }
    return false;
#endif
}

bool tensor_backend_compiled() {
#if VECTOR_DB_V3_CUDA_ENABLED && VECTOR_DB_V3_TENSOR_ENABLED
    return true;
#else
    return false;
#endif
}

bool tensor_backend_available(std::string* reason) {
#if VECTOR_DB_V3_CUDA_ENABLED && VECTOR_DB_V3_TENSOR_ENABLED
    std::string cuda_reason;
    if (!cuda_backend_available(&cuda_reason)) {
        if (reason != nullptr) {
            *reason = cuda_reason;
        }
        return false;
    }
    if (reason != nullptr) {
        *reason = "ok";
    }
    return true;
#else
    if (reason != nullptr) {
        *reason = "tensor_backend_not_compiled";
    }
    return false;
#endif
}

Status run_kmeans_cuda(
    const std::vector<std::vector<float>>& vectors,
    std::uint32_t k,
    std::uint32_t max_iterations,
    PrecisionPreference precision_preference,
    bool tensor_required,
    CudaPipelineContext* pipeline_context,
    KMeansResult* out) {
#if VECTOR_DB_V3_CUDA_ENABLED
    RuntimeInfo info{};
    info.observed = true;
    info.cuda_compiled = cuda_backend_compiled();
    info.tensor_compiled = tensor_backend_compiled();
    info.backend_path = "cuda_fp32";
    info.gpu_arch_class = "unknown";
    if (pipeline_context != nullptr) {
        info.residency = pipeline_context->stats();
    }

    std::string cuda_reason;
    info.cuda_available = cuda_backend_available(&cuda_reason);
    if (!info.cuda_available) {
        info.fallback_reason = cuda_reason;
        set_runtime_info_for_stage(info);
        return Status::Error("kmeans cuda unavailable: " + cuda_reason);
    }
    info.gpu_arch_class = "ampere";

    std::string tensor_reason;
    info.tensor_available = tensor_backend_available(&tensor_reason);
    bool use_tensor = false;

    if (precision_preference == PrecisionPreference::TensorFP16) {
        if (!info.tensor_available) {
            if (tensor_required) {
                info.fallback_reason = tensor_reason;
                set_runtime_info_for_stage(info);
                return Status::Error("kmeans tensor unavailable: " + tensor_reason);
            }
            info.fallback_reason = tensor_reason.empty() ? "tensor_runtime_unavailable" : tensor_reason;
            use_tensor = false;
        } else {
            std::string effective_reason;
            info.tensor_effective = tensor_path_effective(static_cast<std::uint32_t>(vectors.size()), k, kVectorDim, &effective_reason);
            if (!info.tensor_effective) {
                if (tensor_required) {
                    info.fallback_reason = effective_reason;
                    set_runtime_info_for_stage(info);
                    return Status::Error("kmeans tensor unavailable: " + effective_reason);
                }
                info.fallback_reason = effective_reason;
                use_tensor = false;
            } else {
                use_tensor = true;
            }
        }
    } else if (precision_preference == PrecisionPreference::Auto) {
        if (info.tensor_available) {
            std::string effective_reason;
            info.tensor_effective = tensor_path_effective(static_cast<std::uint32_t>(vectors.size()), k, kVectorDim, &effective_reason);
            use_tensor = info.tensor_effective;
            if (!use_tensor) {
                info.fallback_reason = effective_reason;
            }
        } else if (tensor_required) {
            info.fallback_reason = tensor_reason.empty() ? "tensor_runtime_unavailable" : tensor_reason;
            set_runtime_info_for_stage(info);
            return Status::Error("kmeans tensor unavailable: " + info.fallback_reason);
        }
    } else if (precision_preference == PrecisionPreference::FP32 && tensor_required) {
        info.fallback_reason = "tensor_required_but_fp32_requested";
        set_runtime_info_for_stage(info);
        return Status::Error("kmeans tensor unavailable: tensor_required_but_fp32_requested");
    }

    if (use_tensor) {
        Status tensor_st = run_kmeans_cuda_tensor(vectors, k, max_iterations, pipeline_context, out);
        if (!tensor_st.ok) {
            if (tensor_required) {
                info.fallback_reason = "tensor_runtime_unavailable";
                set_runtime_info_for_stage(info);
                return tensor_st;
            }
            Status fp32_st = run_kmeans_cuda_impl(vectors, k, max_iterations, pipeline_context, out);
            if (pipeline_context != nullptr) {
                info.residency = pipeline_context->stats();
            }
            info.tensor_active = false;
            info.backend_path = "cuda_fp32";
            info.fallback_reason = "tensor_runtime_unavailable";
            set_runtime_info_for_stage(info);
            return fp32_st;
        }
        info.tensor_active = true;
        info.backend_path = "cuda_tensor_fp16";
        info.fallback_reason.clear();
        if (pipeline_context != nullptr) {
            info.residency = pipeline_context->stats();
        }
        set_runtime_info_for_stage(info);
        return tensor_st;
    }

    Status fp32_st = run_kmeans_cuda_impl(vectors, k, max_iterations, pipeline_context, out);
    info.tensor_active = false;
    info.backend_path = "cuda_fp32";
    if (pipeline_context != nullptr) {
        info.residency = pipeline_context->stats();
    }
    if (fp32_st.ok) {
        set_runtime_info_for_stage(info);
    } else {
        info.fallback_reason = "cuda_fp32_runtime_failed";
        set_runtime_info_for_stage(info);
    }
    return fp32_st;
#else
    (void)vectors;
    (void)k;
    (void)max_iterations;
    (void)precision_preference;
    (void)tensor_required;
    (void)pipeline_context;
    (void)out;
    return Status::Error("kmeans cuda unavailable: cuda_not_compiled");
#endif
}

}  // namespace vector_db_v3::kmeans
