#include "vector_db/clustering.hpp"

#ifdef VECTOR_DB_USE_CUDA

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    __half* d_vectors = nullptr;
    __half* d_centroids = nullptr;
    float* d_scores = nullptr;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 8 * 1024 * 1024;

    const std::vector<__half> vectors_fp16 = fp32_to_fp16(vectors_row_major);
    const std::vector<__half> centroids_fp16 = fp32_to_fp16(centroids_row_major);
    const std::size_t v_bytes = vectors_fp16.size() * sizeof(__half);
    const std::size_t c_bytes = centroids_fp16.size() * sizeof(__half);
    const std::size_t s_bytes = score_count * sizeof(float);

    if (cudaMalloc(reinterpret_cast<void**>(&d_vectors), v_bytes) != cudaSuccess) {
        return Status::Error("cudaMalloc failed for vectors");
    }
    if (cudaMalloc(reinterpret_cast<void**>(&d_centroids), c_bytes) != cudaSuccess) {
        cudaFree(d_vectors);
        return Status::Error("cudaMalloc failed for centroids");
    }
    if (cudaMalloc(reinterpret_cast<void**>(&d_scores), s_bytes) != cudaSuccess) {
        cudaFree(d_vectors);
        cudaFree(d_centroids);
        return Status::Error("cudaMalloc failed for scores");
    }
    if (cudaMalloc(&workspace, workspace_bytes) != cudaSuccess) {
        workspace = nullptr;
        workspace_bytes = 0;
    }

    cudaMemcpy(d_vectors, vectors_fp16.data(), v_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids_fp16.data(), c_bytes, cudaMemcpyHostToDevice);

    if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_vectors);
        cudaFree(d_centroids);
        cudaFree(d_scores);
        if (workspace != nullptr) {
            cudaFree(workspace);
        }
        return Status::Error("cublasLtCreate failed");
    }

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_T;
    if (cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
        cublasLtDestroy(lt);
        cudaFree(d_vectors);
        cudaFree(d_centroids);
        cudaFree(d_scores);
        if (workspace != nullptr) {
            cudaFree(workspace);
        }
        return Status::Error("cublasLtMatmulDescCreate failed");
    }
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

    if (cublasLtMatrixLayoutCreate(
            &a_desc,
            CUDA_R_16F,
            static_cast<std::uint64_t>(n_vectors),
            static_cast<std::uint64_t>(dim),
            static_cast<std::int64_t>(dim))
            != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(
               &b_desc,
               CUDA_R_16F,
               static_cast<std::uint64_t>(k_centroids),
               static_cast<std::uint64_t>(dim),
               static_cast<std::int64_t>(dim))
               != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(
               &c_desc,
               CUDA_R_32F,
               static_cast<std::uint64_t>(n_vectors),
               static_cast<std::uint64_t>(k_centroids),
               static_cast<std::int64_t>(k_centroids))
               != CUBLAS_STATUS_SUCCESS) {
        if (a_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(a_desc);
        }
        if (b_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(b_desc);
        }
        if (c_desc != nullptr) {
            cublasLtMatrixLayoutDestroy(c_desc);
        }
        cublasLtMatmulDescDestroy(op_desc);
        cublasLtDestroy(lt);
        cudaFree(d_vectors);
        cudaFree(d_centroids);
        cudaFree(d_scores);
        if (workspace != nullptr) {
            cudaFree(workspace);
        }
        return Status::Error("cublasLtMatrixLayoutCreate failed");
    }
    set_row_major(a_desc);
    set_row_major(b_desc);
    set_row_major(c_desc);

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned_results = 0;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatrixLayoutDestroy(b_desc);
        cublasLtMatrixLayoutDestroy(c_desc);
        cublasLtMatmulDescDestroy(op_desc);
        cublasLtDestroy(lt);
        cudaFree(d_vectors);
        cudaFree(d_centroids);
        cudaFree(d_scores);
        if (workspace != nullptr) {
            cudaFree(workspace);
        }
        return Status::Error("cublasLtMatmulPreferenceCreate failed");
    }
    cublasLtMatmulPreferenceSetAttribute(
        pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes,
        sizeof(workspace_bytes));
    if (cublasLtMatmulAlgoGetHeuristic(
            lt,
            op_desc,
            a_desc,
            b_desc,
            c_desc,
            c_desc,
            pref,
            1,
            &heuristic,
            &returned_results)
        != CUBLAS_STATUS_SUCCESS
        || returned_results <= 0) {
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatrixLayoutDestroy(b_desc);
        cublasLtMatrixLayoutDestroy(c_desc);
        cublasLtMatmulDescDestroy(op_desc);
        cublasLtDestroy(lt);
        cudaFree(d_vectors);
        cudaFree(d_centroids);
        cudaFree(d_scores);
        if (workspace != nullptr) {
            cudaFree(workspace);
        }
        return Status::Error("cublasLtMatmulAlgoGetHeuristic failed");
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t matmul_status = cublasLtMatmul(
        lt,
        op_desc,
        &alpha,
        d_vectors,
        a_desc,
        d_centroids,
        b_desc,
        &beta,
        d_scores,
        c_desc,
        d_scores,
        c_desc,
        &heuristic.algo,
        workspace,
        workspace_bytes,
        0);
    if (matmul_status != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatrixLayoutDestroy(b_desc);
        cublasLtMatrixLayoutDestroy(c_desc);
        cublasLtMatmulDescDestroy(op_desc);
        cublasLtDestroy(lt);
        cudaFree(d_vectors);
        cudaFree(d_centroids);
        cudaFree(d_scores);
        if (workspace != nullptr) {
            cudaFree(workspace);
        }
        return Status::Error("cublasLtMatmul failed");
    }

    cudaDeviceSynchronize();
    cudaMemcpy(out_scores_row_major->data(), d_scores, s_bytes, cudaMemcpyDeviceToHost);

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(a_desc);
    cublasLtMatrixLayoutDestroy(b_desc);
    cublasLtMatrixLayoutDestroy(c_desc);
    cublasLtMatmulDescDestroy(op_desc);
    cublasLtDestroy(lt);
    cudaFree(d_vectors);
    cudaFree(d_centroids);
    cudaFree(d_scores);
    if (workspace != nullptr) {
        cudaFree(workspace);
    }
    if (out_tensor_core_enabled != nullptr) {
        *out_tensor_core_enabled = true;
    }
    if (out_backend_name != nullptr) {
        *out_backend_name = "cublaslt";
    }
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

    float* d_scores = nullptr;
    float* d_best = nullptr;
    std::uint32_t* d_labels = nullptr;
    const std::size_t s_bytes = scores_row_major.size() * sizeof(float);
    const std::size_t v_bytes = n_vectors * sizeof(float);
    const std::size_t l_bytes = n_vectors * sizeof(std::uint32_t);
    if (cudaMalloc(reinterpret_cast<void**>(&d_scores), s_bytes) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&d_best), v_bytes) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&d_labels), l_bytes) != cudaSuccess) {
        if (d_scores != nullptr) {
            cudaFree(d_scores);
        }
        if (d_best != nullptr) {
            cudaFree(d_best);
        }
        if (d_labels != nullptr) {
            cudaFree(d_labels);
        }
        return Status::Error("cudaMalloc failed for assign_top1");
    }
    cudaMemcpy(d_scores, scores_row_major.data(), s_bytes, cudaMemcpyHostToDevice);
    const int threads = 256;
    const int blocks = static_cast<int>((n_vectors + static_cast<std::size_t>(threads) - 1U) / static_cast<std::size_t>(threads));
    assign_top1_kernel<<<blocks, threads>>>(d_scores, d_best, d_labels, n_vectors, k_centroids);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_scores);
        cudaFree(d_best);
        cudaFree(d_labels);
        return Status::Error("assign_top1 kernel launch failed");
    }
    cudaDeviceSynchronize();
    cudaMemcpy(out_best_scores->data(), d_best, v_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_labels->data(), d_labels, l_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_scores);
    cudaFree(d_best);
    cudaFree(d_labels);
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

    float* d_vectors = nullptr;
    float* d_sums = nullptr;
    std::uint32_t* d_labels = nullptr;
    std::uint32_t* d_counts = nullptr;
    const std::size_t v_bytes = vectors_row_major.size() * sizeof(float);
    const std::size_t s_bytes = k_centroids * dim * sizeof(float);
    const std::size_t l_bytes = labels.size() * sizeof(std::uint32_t);
    const std::size_t c_bytes = k_centroids * sizeof(std::uint32_t);
    if (cudaMalloc(reinterpret_cast<void**>(&d_vectors), v_bytes) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&d_sums), s_bytes) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&d_labels), l_bytes) != cudaSuccess
        || cudaMalloc(reinterpret_cast<void**>(&d_counts), c_bytes) != cudaSuccess) {
        if (d_vectors != nullptr) {
            cudaFree(d_vectors);
        }
        if (d_sums != nullptr) {
            cudaFree(d_sums);
        }
        if (d_labels != nullptr) {
            cudaFree(d_labels);
        }
        if (d_counts != nullptr) {
            cudaFree(d_counts);
        }
        return Status::Error("cudaMalloc failed for reduce_centroids");
    }
    cudaMemcpy(d_vectors, vectors_row_major.data(), v_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels.data(), l_bytes, cudaMemcpyHostToDevice);

    const int threads = 256;
    const std::size_t total_sum = k_centroids * dim;
    const int clear_blocks = static_cast<int>((std::max(total_sum, k_centroids) + static_cast<std::size_t>(threads) - 1U) / static_cast<std::size_t>(threads));
    clear_centroid_buffers<<<clear_blocks, threads>>>(d_sums, d_counts, total_sum, k_centroids);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_vectors);
        cudaFree(d_sums);
        cudaFree(d_labels);
        cudaFree(d_counts);
        return Status::Error("clear_centroid_buffers launch failed");
    }
    const int reduce_blocks = static_cast<int>((n_vectors + static_cast<std::size_t>(threads) - 1U) / static_cast<std::size_t>(threads));
    reduce_centroids_kernel<<<reduce_blocks, threads>>>(d_vectors, d_labels, d_sums, d_counts, n_vectors, k_centroids, dim);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_vectors);
        cudaFree(d_sums);
        cudaFree(d_labels);
        cudaFree(d_counts);
        return Status::Error("reduce_centroids kernel launch failed");
    }
    const int norm_blocks = static_cast<int>((k_centroids + static_cast<std::size_t>(threads) - 1U) / static_cast<std::size_t>(threads));
    normalize_centroids_kernel<<<norm_blocks, threads>>>(d_sums, d_counts, k_centroids, dim);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_vectors);
        cudaFree(d_sums);
        cudaFree(d_labels);
        cudaFree(d_counts);
        return Status::Error("normalize_centroids kernel launch failed");
    }
    cudaDeviceSynchronize();

    cudaMemcpy(out_centroids_row_major->data(), d_sums, s_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_vectors);
    cudaFree(d_sums);
    cudaFree(d_labels);
    cudaFree(d_counts);
    return Status::Ok();
}

}  // namespace vector_db

#endif

