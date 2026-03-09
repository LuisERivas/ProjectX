#include "vector_db/clustering.hpp"

#ifdef VECTOR_DB_USE_CUDA

#include <cuda_runtime.h>

#include <vector>

namespace vector_db {

namespace {

__global__ void dot_kernel(
    const float* vectors,
    const float* centroids,
    float* out_scores,
    std::size_t n_vectors,
    std::size_t k_centroids,
    std::size_t dim) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    const std::size_t total = n_vectors * k_centroids;
    if (idx >= total) {
        return;
    }
    const std::size_t i = idx / k_centroids;
    const std::size_t c = idx % k_centroids;
    const float* x = vectors + i * dim;
    const float* mu = centroids + c * dim;
    float s = 0.0f;
    for (std::size_t d = 0; d < dim; ++d) {
        s += x[d] * mu[d];
    }
    out_scores[idx] = s;
}

}  // namespace

bool cuda_dot_products_available() { return true; }

Status cuda_compute_dot_products(
    const std::vector<float>& vectors_row_major,
    const std::vector<float>& centroids_row_major,
    std::size_t n_vectors,
    std::size_t k_centroids,
    std::size_t dim,
    std::vector<float>* out_scores_row_major) {
    if (out_scores_row_major == nullptr) {
        return Status::Error("cuda output buffer is null");
    }
    const std::size_t score_count = n_vectors * k_centroids;
    out_scores_row_major->assign(score_count, 0.0f);
    if (vectors_row_major.size() != n_vectors * dim) {
        return Status::Error("cuda vectors buffer shape mismatch");
    }
    if (centroids_row_major.size() != k_centroids * dim) {
        return Status::Error("cuda centroids buffer shape mismatch");
    }

    float* d_vectors = nullptr;
    float* d_centroids = nullptr;
    float* d_scores = nullptr;
    const std::size_t v_bytes = vectors_row_major.size() * sizeof(float);
    const std::size_t c_bytes = centroids_row_major.size() * sizeof(float);
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

    cudaMemcpy(d_vectors, vectors_row_major.data(), v_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids_row_major.data(), c_bytes, cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = static_cast<int>((score_count + static_cast<std::size_t>(threads) - 1U) / static_cast<std::size_t>(threads));
    dot_kernel<<<blocks, threads>>>(d_vectors, d_centroids, d_scores, n_vectors, k_centroids, dim);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_vectors);
        cudaFree(d_centroids);
        cudaFree(d_scores);
        return Status::Error("cuda kernel launch failed");
    }
    cudaDeviceSynchronize();
    cudaMemcpy(out_scores_row_major->data(), d_scores, s_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_vectors);
    cudaFree(d_centroids);
    cudaFree(d_scores);
    return Status::Ok();
}

}  // namespace vector_db

#endif

