#include "kmeans_kernels.hpp"

#include <cuda_runtime.h>

namespace vector_db_v3::kmeans::cuda {

namespace {

__global__ void kmeans_assign_kernel(
    const float* vectors,
    const float* centroids,
    std::uint32_t* assignments,
    float* min_dists,
    std::uint32_t vector_count,
    std::uint32_t k,
    std::size_t dim) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vector_count) {
        return;
    }

    const float* vec = vectors + static_cast<std::size_t>(idx) * dim;
    std::uint32_t best_idx = 0U;
    // Avoid toolkit-version-specific infinity macros in device code.
    float best_dist = 3.402823466e+38F;
    for (std::uint32_t c = 0; c < k; ++c) {
        const float* centroid = centroids + static_cast<std::size_t>(c) * dim;
        float dist = 0.0f;
        for (std::size_t d = 0; d < dim; ++d) {
            const float delta = vec[d] - centroid[d];
            dist += delta * delta;
        }
        // Keep the first centroid in equal-distance ties.
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = c;
        }
    }
    assignments[idx] = best_idx;
    min_dists[idx] = best_dist;
}

__global__ void kmeans_accumulate_kernel(
    const float* vectors,
    const std::uint32_t* assignments,
    float* sums,
    std::uint32_t* counts,
    std::uint32_t vector_count,
    std::uint32_t k,
    std::size_t dim) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vector_count) {
        return;
    }
    const std::uint32_t bucket = assignments[idx];
    if (bucket >= k) {
        return;
    }
    atomicAdd(counts + bucket, 1U);
    const float* vec = vectors + static_cast<std::size_t>(idx) * dim;
    float* sum_row = sums + static_cast<std::size_t>(bucket) * dim;
    for (std::size_t d = 0; d < dim; ++d) {
        atomicAdd(sum_row + d, vec[d]);
    }
}

__global__ void kmeans_update_kernel(
    float* centroids,
    const float* sums,
    const std::uint32_t* counts,
    std::uint32_t k,
    std::size_t dim) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(k) * dim;
    if (idx >= total) {
        return;
    }
    const std::uint32_t c = static_cast<std::uint32_t>(idx / dim);
    const std::size_t d = idx % dim;
    const std::uint32_t count = counts[c];
    if (count == 0U) {
        return;
    }
    centroids[idx] = sums[idx] / static_cast<float>(count);
}

__global__ void kmeans_objective_kernel(
    const float* vectors,
    const float* centroids,
    const std::uint32_t* assignments,
    float* objective,
    std::uint32_t vector_count,
    std::size_t dim) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vector_count) {
        return;
    }
    const std::uint32_t c = assignments[idx];
    const float* vec = vectors + static_cast<std::size_t>(idx) * dim;
    const float* centroid = centroids + static_cast<std::size_t>(c) * dim;
    float dist = 0.0f;
    for (std::size_t d = 0; d < dim; ++d) {
        const float delta = vec[d] - centroid[d];
        dist += delta * delta;
    }
    objective[idx] = dist;
}

bool launch_ok() {
    const cudaError_t st = cudaGetLastError();
    return st == cudaSuccess;
}

}  // namespace

bool launch_assignment_kernel(
    const float* d_vectors,
    const float* d_centroids,
    std::uint32_t* d_assignments,
    float* d_min_dists,
    std::uint32_t vector_count,
    std::uint32_t k,
    std::size_t dim) {
    constexpr std::uint32_t kBlock = 128U;
    const std::uint32_t grid = (vector_count + kBlock - 1U) / kBlock;
    kmeans_assign_kernel<<<grid, kBlock>>>(d_vectors, d_centroids, d_assignments, d_min_dists, vector_count, k, dim);
    return launch_ok();
}

bool launch_accumulate_kernel(
    const float* d_vectors,
    const std::uint32_t* d_assignments,
    float* d_sums,
    std::uint32_t* d_counts,
    std::uint32_t vector_count,
    std::uint32_t k,
    std::size_t dim) {
    constexpr std::uint32_t kBlock = 128U;
    const std::uint32_t grid = (vector_count + kBlock - 1U) / kBlock;
    kmeans_accumulate_kernel<<<grid, kBlock>>>(d_vectors, d_assignments, d_sums, d_counts, vector_count, k, dim);
    return launch_ok();
}

bool launch_update_kernel(
    float* d_centroids,
    const float* d_sums,
    const std::uint32_t* d_counts,
    std::uint32_t k,
    std::size_t dim) {
    constexpr std::uint32_t kBlock = 128U;
    const std::uint32_t total = static_cast<std::uint32_t>(k * dim);
    const std::uint32_t grid = (total + kBlock - 1U) / kBlock;
    kmeans_update_kernel<<<grid, kBlock>>>(d_centroids, d_sums, d_counts, k, dim);
    return launch_ok();
}

bool launch_objective_kernel(
    const float* d_vectors,
    const float* d_centroids,
    const std::uint32_t* d_assignments,
    float* d_objective,
    std::uint32_t vector_count,
    std::size_t dim) {
    constexpr std::uint32_t kBlock = 128U;
    const std::uint32_t grid = (vector_count + kBlock - 1U) / kBlock;
    kmeans_objective_kernel<<<grid, kBlock>>>(d_vectors, d_centroids, d_assignments, d_objective, vector_count, dim);
    return launch_ok();
}

}  // namespace vector_db_v3::kmeans::cuda
