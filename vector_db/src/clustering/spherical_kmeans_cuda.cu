#include "vector_db/clustering.hpp"

#ifdef VECTOR_DB_USE_CUDA

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>

namespace vector_db {

namespace {

__global__ void assign_top1_kernel(
    const float* scores,
    float* out_scores,
    std::uint32_t* out_labels,
    std::size_t n_vectors,
    std::size_t k_centroids) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (i >= n_vectors) {
        return;
    }
    const float* row = scores + i * k_centroids;
    std::size_t best = 0;
    float best_score = row[0];
    for (std::size_t c = 1; c < k_centroids; ++c) {
        const float s = row[c];
        if (s > best_score) {
            best_score = s;
            best = c;
        }
    }
    out_labels[i] = static_cast<std::uint32_t>(best);
    out_scores[i] = best_score;
}

__global__ void assign_topm_kernel(
    const float* scores,
    std::uint32_t* out_topm,
    std::size_t n_vectors,
    std::size_t k_centroids,
    std::size_t top_m) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (i >= n_vectors) {
        return;
    }
    const float* row = scores + i * k_centroids;
    for (std::size_t m = 0; m < top_m; ++m) {
        float best_score = -CUDART_INF_F;
        std::uint32_t best_idx = 0U;
        for (std::size_t c = 0; c < k_centroids; ++c) {
            const float s = row[c];
            bool already = false;
            for (std::size_t p = 0; p < m; ++p) {
                if (out_topm[i * top_m + p] == static_cast<std::uint32_t>(c)) {
                    already = true;
                    break;
                }
            }
            if (already) {
                continue;
            }
            if (s > best_score) {
                best_score = s;
                best_idx = static_cast<std::uint32_t>(c);
            }
        }
        out_topm[i * top_m + m] = best_idx;
    }
}

__global__ void clear_centroid_buffers(float* sums, std::uint32_t* counts, std::size_t total_sum, std::size_t k_centroids) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (idx < total_sum) {
        sums[idx] = 0.0f;
    }
    if (idx < k_centroids) {
        counts[idx] = 0;
    }
}

__global__ void reduce_centroids_kernel(
    const float* vectors,
    const std::uint32_t* labels,
    float* sums,
    std::uint32_t* counts,
    std::size_t n_vectors,
    std::size_t k_centroids,
    std::size_t dim) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (i >= n_vectors) {
        return;
    }
    const std::uint32_t c = labels[i];
    if (c >= k_centroids) {
        return;
    }
    atomicAdd(&counts[c], 1U);
    const float* x = vectors + i * dim;
    float* sum_row = sums + static_cast<std::size_t>(c) * dim;
    for (std::size_t d = 0; d < dim; ++d) {
        atomicAdd(&sum_row[d], x[d]);
    }
}

__global__ void reduce_centroids_partial_kernel(
    const float* vectors,
    const std::uint32_t* labels,
    float* partial_sums,
    std::uint32_t* partial_counts,
    std::size_t n_vectors,
    std::size_t k_centroids,
    std::size_t dim) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (i >= n_vectors) {
        return;
    }
    const std::size_t block_slot = static_cast<std::size_t>(blockIdx.x);
    const std::uint32_t c = labels[i];
    if (c >= k_centroids) {
        return;
    }
    std::uint32_t* counts_row = partial_counts + block_slot * k_centroids;
    atomicAdd(&counts_row[c], 1U);
    const float* x = vectors + i * dim;
    float* sums_row = partial_sums + (block_slot * k_centroids + static_cast<std::size_t>(c)) * dim;
    for (std::size_t d = 0; d < dim; ++d) {
        atomicAdd(&sums_row[d], x[d]);
    }
}

__global__ void finalize_partial_centroids_kernel(
    const float* partial_sums,
    const std::uint32_t* partial_counts,
    float* out_centroids,
    std::uint32_t* out_counts,
    std::size_t n_blocks,
    std::size_t k_centroids,
    std::size_t dim) {
    const std::size_t c = static_cast<std::size_t>(blockIdx.x);
    const std::size_t d = static_cast<std::size_t>(threadIdx.x);
    if (c >= k_centroids || d >= dim) {
        return;
    }
    float sum = 0.0f;
    std::uint32_t cnt = 0U;
    for (std::size_t b = 0; b < n_blocks; ++b) {
        sum += partial_sums[(b * k_centroids + c) * dim + d];
        cnt += (d == 0) ? partial_counts[b * k_centroids + c] : 0U;
    }
    if (d == 0) {
        out_counts[c] = cnt;
    }
    out_centroids[c * dim + d] = (cnt == 0U) ? 0.0f : sum;
}

__global__ void normalize_centroids_kernel(
    float* centroids,
    const std::uint32_t* counts,
    std::size_t k_centroids,
    std::size_t dim) {
    const std::size_t c = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    if (c >= k_centroids) {
        return;
    }
    const std::uint32_t cnt = counts[c];
    if (cnt == 0) {
        return;
    }
    float* row = centroids + c * dim;
    float norm_sq = 0.0f;
    for (std::size_t d = 0; d < dim; ++d) {
        row[d] = row[d] / static_cast<float>(cnt);
        norm_sq += row[d] * row[d];
    }
    const float norm = sqrtf(fmaxf(norm_sq, 1e-12f));
    for (std::size_t d = 0; d < dim; ++d) {
        row[d] = row[d] / norm;
    }
}

std::vector<__half> fp32_to_fp16(const std::vector<float>& in) {
    std::vector<__half> out(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        out[i] = __float2half(in[i]);
    }
    return out;
}

bool set_row_major(cublasLtMatrixLayout_t layout) {
    const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    return cublasLtMatrixLayoutSetAttribute(
               layout,
               CUBLASLT_MATRIX_LAYOUT_ORDER,
               &order,
               sizeof(order))
        == CUBLAS_STATUS_SUCCESS;
}

struct GpuContextCache {
    __half* d_vectors_fp16 = nullptr;
    __half* d_centroids_fp16 = nullptr;
    float* d_scores = nullptr;
    float* d_vectors_fp32 = nullptr;
    std::uint32_t* d_labels = nullptr;
    float* d_best_scores = nullptr;
    std::uint32_t* d_topm = nullptr;
    float* d_partial_sums = nullptr;
    std::uint32_t* d_partial_counts = nullptr;
    float* d_centroids = nullptr;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 8 * 1024 * 1024;
    std::size_t cap_n = 0;
    std::size_t cap_k = 0;
    std::size_t cap_dim = 0;
    std::size_t cap_topm = 0;
    std::size_t cap_blocks = 0;
    const float* last_vectors_host_ptr = nullptr;
    std::size_t last_vectors_host_count = 0;
    bool vectors_fp16_loaded = false;
    bool vectors_fp32_loaded = false;
    cublasLtHandle_t lt = nullptr;
    bool initialized = false;
};

GpuContextCache& gpu_cache() {
    static GpuContextCache cache;
    return cache;
}

void free_cache_buffers(GpuContextCache& c) {
    if (c.d_vectors_fp16 != nullptr) cudaFree(c.d_vectors_fp16);
    if (c.d_centroids_fp16 != nullptr) cudaFree(c.d_centroids_fp16);
    if (c.d_scores != nullptr) cudaFree(c.d_scores);
    if (c.d_vectors_fp32 != nullptr) cudaFree(c.d_vectors_fp32);
    if (c.d_labels != nullptr) cudaFree(c.d_labels);
    if (c.d_best_scores != nullptr) cudaFree(c.d_best_scores);
    if (c.d_topm != nullptr) cudaFree(c.d_topm);
    if (c.d_partial_sums != nullptr) cudaFree(c.d_partial_sums);
    if (c.d_partial_counts != nullptr) cudaFree(c.d_partial_counts);
    if (c.d_centroids != nullptr) cudaFree(c.d_centroids);
    if (c.workspace != nullptr) cudaFree(c.workspace);
    if (c.lt != nullptr) cublasLtDestroy(c.lt);
    c = GpuContextCache{};
}

Status ensure_cache(
    GpuContextCache& c,
    std::size_t n_vectors,
    std::size_t k_centroids,
    std::size_t dim,
    std::size_t top_m,
    std::size_t n_blocks) {
    if (c.cap_n >= n_vectors && c.cap_k >= k_centroids && c.cap_dim >= dim && c.cap_topm >= top_m && c.cap_blocks >= n_blocks && c.initialized) {
        return Status::Ok();
    }
    free_cache_buffers(c);
    c.cap_n = n_vectors;
    c.cap_k = k_centroids;
    c.cap_dim = dim;
    c.cap_topm = top_m;
    c.cap_blocks = n_blocks;
    const std::size_t vectors_fp16_bytes = n_vectors * dim * sizeof(__half);
    const std::size_t centroids_fp16_bytes = k_centroids * dim * sizeof(__half);
    const std::size_t scores_bytes = n_vectors * k_centroids * sizeof(float);
    const std::size_t vectors_fp32_bytes = n_vectors * dim * sizeof(float);
    const std::size_t labels_bytes = n_vectors * sizeof(std::uint32_t);
    const std::size_t best_scores_bytes = n_vectors * sizeof(float);
    const std::size_t topm_bytes = n_vectors * std::max<std::size_t>(top_m, 1) * sizeof(std::uint32_t);
    const std::size_t partial_sums_bytes = n_blocks * k_centroids * dim * sizeof(float);
    const std::size_t partial_counts_bytes = n_blocks * k_centroids * sizeof(std::uint32_t);
    const std::size_t centroids_bytes = k_centroids * dim * sizeof(float);
    if (cudaMalloc(reinterpret_cast<void**>(&c.d_vectors_fp16), vectors_fp16_bytes) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&c.d_centroids_fp16), centroids_fp16_bytes) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&c.d_scores), scores_bytes) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&c.d_vectors_fp32), vectors_fp32_bytes) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&c.d_labels), labels_bytes) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&c.d_best_scores), best_scores_bytes) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&c.d_topm), topm_bytes) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&c.d_partial_sums), partial_sums_bytes) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&c.d_partial_counts), partial_counts_bytes) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&c.d_centroids), centroids_bytes) != cudaSuccess) {
        free_cache_buffers(c);
        return Status::Error("cudaMalloc failed while creating cache");
    }
    if (cudaMalloc(&c.workspace, c.workspace_bytes) != cudaSuccess) {
        c.workspace = nullptr;
        c.workspace_bytes = 0;
    }
    if (cublasLtCreate(&c.lt) != CUBLAS_STATUS_SUCCESS) {
        free_cache_buffers(c);
        return Status::Error("cublasLtCreate failed");
    }
    c.initialized = true;
    return Status::Ok();
}

Status upload_vectors_if_needed(
    GpuContextCache& c,
    const std::vector<float>& vectors_row_major,
    std::size_t n_vectors,
    std::size_t dim,
    bool need_fp16,
    bool need_fp32) {
    const float* host_ptr = vectors_row_major.empty() ? nullptr : vectors_row_major.data();
    const std::size_t host_count = vectors_row_major.size();
    const bool changed = (host_ptr != c.last_vectors_host_ptr) || (host_count != c.last_vectors_host_count);
    if (changed) {
        c.vectors_fp16_loaded = false;
        c.vectors_fp32_loaded = false;
    }
    if (need_fp32 && !c.vectors_fp32_loaded) {
        const std::size_t bytes = n_vectors * dim * sizeof(float);
        cudaMemcpy(c.d_vectors_fp32, vectors_row_major.data(), bytes, cudaMemcpyHostToDevice);
        c.vectors_fp32_loaded = true;
    }
    if (need_fp16 && !c.vectors_fp16_loaded) {
        const std::vector<__half> vectors_fp16 = fp32_to_fp16(vectors_row_major);
        const std::size_t bytes = n_vectors * dim * sizeof(__half);
        cudaMemcpy(c.d_vectors_fp16, vectors_fp16.data(), bytes, cudaMemcpyHostToDevice);
        c.vectors_fp16_loaded = true;
    }
    c.last_vectors_host_ptr = host_ptr;
    c.last_vectors_host_count = host_count;
    return Status::Ok();
}

Status upload_centroids_fp16(
    GpuContextCache& c,
    const std::vector<float>& centroids_row_major,
    std::size_t k_centroids,
    std::size_t dim) {
    const std::vector<__half> centroids_fp16 = fp32_to_fp16(centroids_row_major);
    const std::size_t bytes = k_centroids * dim * sizeof(__half);
    cudaMemcpy(c.d_centroids_fp16, centroids_fp16.data(), bytes, cudaMemcpyHostToDevice);
    return Status::Ok();
}

Status matmul_scores_cached(
    GpuContextCache& c,
    std::size_t n_vectors,
    std::size_t k_centroids,
    std::size_t dim,
    bool* out_tensor_core_enabled,
    std::string* out_backend_name) {
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_T;
    if (cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
        return Status::Error("cublasLtMatmulDescCreate failed");
    }
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
    if (cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, static_cast<std::uint64_t>(n_vectors), static_cast<std::uint64_t>(dim), static_cast<std::int64_t>(dim)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, static_cast<std::uint64_t>(k_centroids), static_cast<std::uint64_t>(dim), static_cast<std::int64_t>(dim)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, static_cast<std::uint64_t>(n_vectors), static_cast<std::uint64_t>(k_centroids), static_cast<std::int64_t>(k_centroids)) != CUBLAS_STATUS_SUCCESS) {
        if (a_desc != nullptr) cublasLtMatrixLayoutDestroy(a_desc);
        if (b_desc != nullptr) cublasLtMatrixLayoutDestroy(b_desc);
        if (c_desc != nullptr) cublasLtMatrixLayoutDestroy(c_desc);
        cublasLtMatmulDescDestroy(op_desc);
        return Status::Error("cublasLtMatrixLayoutCreate failed");
    }
    set_row_major(a_desc);
    set_row_major(b_desc);
    set_row_major(c_desc);
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatrixLayoutDestroy(b_desc);
        cublasLtMatrixLayoutDestroy(c_desc);
        cublasLtMatmulDescDestroy(op_desc);
        return Status::Error("cublasLtMatmulPreferenceCreate failed");
    }
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &c.workspace_bytes, sizeof(c.workspace_bytes));
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned_results = 0;
    if (cublasLtMatmulAlgoGetHeuristic(c.lt, op_desc, a_desc, b_desc, c_desc, c_desc, pref, 1, &heuristic, &returned_results) != CUBLAS_STATUS_SUCCESS
        || returned_results <= 0) {
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatrixLayoutDestroy(b_desc);
        cublasLtMatrixLayoutDestroy(c_desc);
        cublasLtMatmulDescDestroy(op_desc);
        return Status::Error("cublasLtMatmulAlgoGetHeuristic failed");
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t st = cublasLtMatmul(
        c.lt,
        op_desc,
        &alpha,
        c.d_vectors_fp16,
        a_desc,
        c.d_centroids_fp16,
        b_desc,
        &beta,
        c.d_scores,
        c_desc,
        c.d_scores,
        c_desc,
        &heuristic.algo,
        c.workspace,
        c.workspace_bytes,
        0);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(a_desc);
    cublasLtMatrixLayoutDestroy(b_desc);
    cublasLtMatrixLayoutDestroy(c_desc);
    cublasLtMatmulDescDestroy(op_desc);
    if (st != CUBLAS_STATUS_SUCCESS) {
        return Status::Error("cublasLtMatmul failed");
    }
    if (out_tensor_core_enabled != nullptr) {
        *out_tensor_core_enabled = true;
    }
    if (out_backend_name != nullptr) {
        *out_backend_name = "cublaslt";
    }
    return Status::Ok();
}

Status reduce_centroids_from_device_labels(
    GpuContextCache& c,
    std::size_t n_vectors,
    std::size_t k_centroids,
    std::size_t dim,
    std::size_t n_blocks) {
    const int threads = 256;
    const std::size_t partial_sums_bytes = n_blocks * k_centroids * dim * sizeof(float);
    const std::size_t partial_counts_bytes = n_blocks * k_centroids * sizeof(std::uint32_t);
    const std::size_t count_bytes = k_centroids * sizeof(std::uint32_t);
    cudaMemset(c.d_partial_sums, 0, partial_sums_bytes);
    cudaMemset(c.d_partial_counts, 0, partial_counts_bytes);
    cudaMemset(c.d_best_scores, 0, count_bytes);

    reduce_centroids_partial_kernel<<<static_cast<int>(n_blocks), threads>>>(
        c.d_vectors_fp32,
        c.d_labels,
        c.d_partial_sums,
        c.d_partial_counts,
        n_vectors,
        k_centroids,
        dim);
    if (cudaGetLastError() != cudaSuccess) {
        return Status::Error("reduce_centroids_partial kernel launch failed");
    }

    finalize_partial_centroids_kernel<<<static_cast<int>(k_centroids), static_cast<int>(dim)>>>(
        c.d_partial_sums,
        c.d_partial_counts,
        c.d_centroids,
        reinterpret_cast<std::uint32_t*>(c.d_best_scores),
        n_blocks,
        k_centroids,
        dim);
    if (cudaGetLastError() != cudaSuccess) {
        return Status::Error("finalize_partial_centroids kernel launch failed");
    }

    normalize_centroids_kernel<<<static_cast<int>((k_centroids + static_cast<std::size_t>(threads) - 1U) / static_cast<std::size_t>(threads)), threads>>>(
        c.d_centroids,
        reinterpret_cast<std::uint32_t*>(c.d_best_scores),
        k_centroids,
        dim);
    if (cudaGetLastError() != cudaSuccess) {
        return Status::Error("normalize_centroids kernel launch failed");
    }
    return Status::Ok();
}

}  // namespace

bool cuda_dot_products_available() { return true; }
bool cuda_assignment_kernels_available() { return true; }

Status cuda_compute_dot_products(
    const std::vector<float>& vectors_row_major,
    const std::vector<float>& centroids_row_major,
    std::size_t n_vectors,
    std::size_t k_centroids,
    std::size_t dim,
    std::vector<float>* out_scores_row_major,
    bool* out_tensor_core_enabled,
    std::string* out_backend_name) {
    if (out_scores_row_major == nullptr) {
        return Status::Error("cuda output buffer is null");
    }
    if (out_tensor_core_enabled != nullptr) {
        *out_tensor_core_enabled = false;
    }
    if (out_backend_name != nullptr) {
        *out_backend_name = "cuda_kernel";
    }
    const std::size_t score_count = n_vectors * k_centroids;
    out_scores_row_major->assign(score_count, 0.0f);
    if (vectors_row_major.size() != n_vectors * dim) {
        return Status::Error("cuda vectors buffer shape mismatch");
    }
    if (centroids_row_major.size() != k_centroids * dim) {
        return Status::Error("cuda centroids buffer shape mismatch");
    }

    const int threads = 256;
    const std::size_t n_blocks = (n_vectors + static_cast<std::size_t>(threads) - 1U) / static_cast<std::size_t>(threads);
    auto& cache = gpu_cache();
    if (const Status s = ensure_cache(cache, n_vectors, k_centroids, dim, 1, n_blocks); !s.ok) {
        return s;
    }
    const std::size_t s_bytes = score_count * sizeof(float);
    if (const Status s = upload_vectors_if_needed(cache, vectors_row_major, n_vectors, dim, true, false); !s.ok) {
        return s;
    }
    if (const Status s = upload_centroids_fp16(cache, centroids_row_major, k_centroids, dim); !s.ok) {
        return s;
    }
    if (const Status s = matmul_scores_cached(cache, n_vectors, k_centroids, dim, out_tensor_core_enabled, out_backend_name); !s.ok) {
        return s;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(out_scores_row_major->data(), cache.d_scores, s_bytes, cudaMemcpyDeviceToHost);
    return Status::Ok();
}

Status cuda_assign_top1_labels(
    const std::vector<float>& scores_row_major,
    std::size_t n_vectors,
    std::size_t k_centroids,
    std::vector<std::uint32_t>* out_labels,
    std::vector<float>* out_best_scores) {
    if (out_labels == nullptr || out_best_scores == nullptr) {
        return Status::Error("assign_top1 outputs are null");
    }
    out_labels->assign(n_vectors, 0U);
    out_best_scores->assign(n_vectors, 0.0f);
    if (scores_row_major.size() != n_vectors * k_centroids) {
        return Status::Error("assign_top1 score matrix shape mismatch");
    }

    const int threads = 256;
    const std::size_t n_blocks = (n_vectors + static_cast<std::size_t>(threads) - 1U) / static_cast<std::size_t>(threads);
    auto& cache = gpu_cache();
    if (const Status s = ensure_cache(cache, n_vectors, k_centroids, 1, 1, n_blocks); !s.ok) {
        return s;
    }
    const std::size_t s_bytes = scores_row_major.size() * sizeof(float);
    const std::size_t v_bytes = n_vectors * sizeof(float);
    const std::size_t l_bytes = n_vectors * sizeof(std::uint32_t);
    cudaMemcpy(cache.d_scores, scores_row_major.data(), s_bytes, cudaMemcpyHostToDevice);
    assign_top1_kernel<<<static_cast<int>(n_blocks), threads>>>(cache.d_scores, cache.d_best_scores, cache.d_labels, n_vectors, k_centroids);
    if (cudaGetLastError() != cudaSuccess) {
        return Status::Error("assign_top1 kernel launch failed");
    }
    cudaDeviceSynchronize();
    cudaMemcpy(out_best_scores->data(), cache.d_best_scores, v_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_labels->data(), cache.d_labels, l_bytes, cudaMemcpyDeviceToHost);
    return Status::Ok();
}

Status cuda_reduce_centroids_top1(
    const std::vector<float>& vectors_row_major,
    const std::vector<std::uint32_t>& labels,
    std::size_t n_vectors,
    std::size_t k_centroids,
    std::size_t dim,
    std::vector<float>* out_centroids_row_major) {
    if (out_centroids_row_major == nullptr) {
        return Status::Error("reduce_centroids output is null");
    }
    if (vectors_row_major.size() != n_vectors * dim || labels.size() != n_vectors) {
        return Status::Error("reduce_centroids input shape mismatch");
    }
    out_centroids_row_major->assign(k_centroids * dim, 0.0f);

    const int threads = 256;
    const std::size_t n_blocks = (n_vectors + static_cast<std::size_t>(threads) - 1U) / static_cast<std::size_t>(threads);
    auto& cache = gpu_cache();
    if (const Status s = ensure_cache(cache, n_vectors, k_centroids, dim, 1, n_blocks); !s.ok) {
        return s;
    }
    const std::size_t l_bytes = labels.size() * sizeof(std::uint32_t);
    const std::size_t s_bytes = k_centroids * dim * sizeof(float);
    if (const Status s = upload_vectors_if_needed(cache, vectors_row_major, n_vectors, dim, false, true); !s.ok) {
        return s;
    }
    cudaMemcpy(cache.d_labels, labels.data(), l_bytes, cudaMemcpyHostToDevice);
    if (const Status s = reduce_centroids_from_device_labels(cache, n_vectors, k_centroids, dim, n_blocks); !s.ok) {
        return s;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(out_centroids_row_major->data(), cache.d_centroids, s_bytes, cudaMemcpyDeviceToHost);
    return Status::Ok();
}

Status cuda_kmeans_iteration_top1(
    const std::vector<float>& vectors_row_major,
    const std::vector<float>& centroids_row_major,
    std::size_t n_vectors,
    std::size_t k_centroids,
    std::size_t dim,
    std::vector<float>* out_centroids_row_major,
    std::vector<std::uint32_t>* out_labels,
    std::vector<float>* out_best_scores,
    bool* out_tensor_core_enabled,
    std::string* out_backend_name,
    double* out_scoring_ms) {
    if (out_centroids_row_major == nullptr || out_labels == nullptr || out_best_scores == nullptr) {
        return Status::Error("null output pointers for cuda_kmeans_iteration_top1");
    }
    if (vectors_row_major.size() != n_vectors * dim || centroids_row_major.size() != k_centroids * dim) {
        return Status::Error("cuda_kmeans_iteration_top1 input shape mismatch");
    }
    const int threads = 256;
    const std::size_t n_blocks = (n_vectors + static_cast<std::size_t>(threads) - 1U) / static_cast<std::size_t>(threads);
    auto& cache = gpu_cache();
    if (const Status s = ensure_cache(cache, n_vectors, k_centroids, dim, 1, n_blocks); !s.ok) {
        return s;
    }
    if (const Status s = upload_vectors_if_needed(cache, vectors_row_major, n_vectors, dim, true, true); !s.ok) {
        return s;
    }
    if (const Status s = upload_centroids_fp16(cache, centroids_row_major, k_centroids, dim); !s.ok) {
        return s;
    }
    auto t0 = std::chrono::steady_clock::now();
    if (const Status s = matmul_scores_cached(cache, n_vectors, k_centroids, dim, out_tensor_core_enabled, out_backend_name); !s.ok) {
        return s;
    }
    assign_top1_kernel<<<static_cast<int>(n_blocks), threads>>>(
        cache.d_scores,
        cache.d_best_scores,
        cache.d_labels,
        n_vectors,
        k_centroids);
    if (cudaGetLastError() != cudaSuccess) {
        return Status::Error("assign_top1 kernel launch failed");
    }
    if (const Status s = reduce_centroids_from_device_labels(cache, n_vectors, k_centroids, dim, n_blocks); !s.ok) {
        return s;
    }
    cudaDeviceSynchronize();
    out_centroids_row_major->assign(k_centroids * dim, 0.0f);
    out_labels->assign(n_vectors, 0U);
    out_best_scores->assign(n_vectors, 0.0f);
    const std::size_t cent_bytes = k_centroids * dim * sizeof(float);
    const std::size_t lbl_bytes = n_vectors * sizeof(std::uint32_t);
    const std::size_t best_bytes = n_vectors * sizeof(float);
    cudaMemcpy(out_centroids_row_major->data(), cache.d_centroids, cent_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_labels->data(), cache.d_labels, lbl_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_best_scores->data(), cache.d_best_scores, best_bytes, cudaMemcpyDeviceToHost);
    auto t1 = std::chrono::steady_clock::now();
    if (out_scoring_ms != nullptr) {
        *out_scoring_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return Status::Ok();
}

Status cuda_topm_from_centroids(
    const std::vector<float>& vectors_row_major,
    const std::vector<float>& centroids_row_major,
    std::size_t n_vectors,
    std::size_t k_centroids,
    std::size_t dim,
    std::size_t top_m,
    std::vector<std::vector<std::uint32_t>>* out_top_m,
    std::vector<float>* out_scores_row_major,
    bool* out_tensor_core_enabled,
    std::string* out_backend_name,
    double* out_scoring_ms) {
    if (out_top_m == nullptr || out_scores_row_major == nullptr) {
        return Status::Error("null outputs for cuda_topm_from_centroids");
    }
    if (vectors_row_major.size() != n_vectors * dim || centroids_row_major.size() != k_centroids * dim) {
        return Status::Error("cuda_topm_from_centroids input shape mismatch");
    }
    const int threads = 256;
    const std::size_t n_blocks = (n_vectors + static_cast<std::size_t>(threads) - 1U) / static_cast<std::size_t>(threads);
    auto& cache = gpu_cache();
    if (const Status s = ensure_cache(cache, n_vectors, k_centroids, dim, std::max<std::size_t>(top_m, 1), n_blocks); !s.ok) {
        return s;
    }
    if (const Status s = upload_vectors_if_needed(cache, vectors_row_major, n_vectors, dim, true, false); !s.ok) {
        return s;
    }
    if (const Status s = upload_centroids_fp16(cache, centroids_row_major, k_centroids, dim); !s.ok) {
        return s;
    }
    auto t0 = std::chrono::steady_clock::now();
    if (const Status s = matmul_scores_cached(cache, n_vectors, k_centroids, dim, out_tensor_core_enabled, out_backend_name); !s.ok) {
        return s;
    }
    assign_topm_kernel<<<static_cast<int>(n_blocks), threads>>>(
        cache.d_scores,
        cache.d_topm,
        n_vectors,
        k_centroids,
        std::max<std::size_t>(top_m, 1));
    if (cudaGetLastError() != cudaSuccess) {
        return Status::Error("assign_topm kernel launch failed");
    }
    cudaDeviceSynchronize();
    out_scores_row_major->assign(n_vectors * k_centroids, 0.0f);
    const std::size_t s_bytes = out_scores_row_major->size() * sizeof(float);
    cudaMemcpy(out_scores_row_major->data(), cache.d_scores, s_bytes, cudaMemcpyDeviceToHost);
    std::vector<std::uint32_t> flat(n_vectors * std::max<std::size_t>(top_m, 1), 0U);
    cudaMemcpy(flat.data(), cache.d_topm, flat.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    out_top_m->assign(n_vectors, {});
    for (std::size_t i = 0; i < n_vectors; ++i) {
        auto& row = (*out_top_m)[i];
        row.reserve(top_m);
        for (std::size_t m = 0; m < top_m; ++m) {
            row.push_back(flat[i * std::max<std::size_t>(top_m, 1) + m]);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    if (out_scoring_ms != nullptr) {
        *out_scoring_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return Status::Ok();
}

}  // namespace vector_db

#endif

