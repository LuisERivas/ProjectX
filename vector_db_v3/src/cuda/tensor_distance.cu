#include "tensor_distance.hpp"

#include <cuda_runtime.h>

namespace vector_db_v3::kmeans::cuda {

namespace {

__global__ void pack_rowmajor_to_colmajor_half_kernel(
    const float* src_rowmajor,
    __half* dst_colmajor,
    std::uint32_t rows,
    std::size_t dim) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * dim;
    if (idx >= total) {
        return;
    }
    const std::uint32_t row = static_cast<std::uint32_t>(idx / dim);
    const std::size_t col = idx % dim;
    const float value = src_rowmajor[static_cast<std::size_t>(row) * dim + col];
    dst_colmajor[col + static_cast<std::size_t>(row) * dim] = __float2half(value);
}

__global__ void row_norms_f32_kernel(
    const float* matrix_rowmajor,
    float* norms,
    std::uint32_t rows,
    std::size_t dim) {
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float* base = matrix_rowmajor + static_cast<std::size_t>(row) * dim;
    float out = 0.0f;
    for (std::size_t d = 0; d < dim; ++d) {
        const float v = base[d];
        out += v * v;
    }
    norms[row] = out;
}

__global__ void argmin_from_dot_kn_kernel(
    const float* dot_kn,
    const float* vector_norms,
    const float* centroid_norms,
    std::uint32_t* assignments,
    float* min_dists,
    std::uint32_t n,
    std::uint32_t k) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    std::uint32_t best = 0U;
    float best_dist = 3.402823466e+38F;
    for (std::uint32_t c = 0; c < k; ++c) {
        const float dot = dot_kn[c + static_cast<std::size_t>(idx) * k];
        const float dist = vector_norms[idx] + centroid_norms[c] - (2.0f * dot);
        if (dist < best_dist) {
            best_dist = dist;
            best = c;
        }
    }
    assignments[idx] = best;
    min_dists[idx] = best_dist;
}

bool launch_ok() {
    const cudaError_t st = cudaGetLastError();
    return st == cudaSuccess;
}

}  // namespace

bool launch_pack_rowmajor_to_colmajor_half(
    const float* d_src_rowmajor,
    __half* d_dst_colmajor,
    std::uint32_t rows,
    std::size_t dim) {
    constexpr std::uint32_t kBlock = 128U;
    const std::uint32_t total = static_cast<std::uint32_t>(rows * dim);
    const std::uint32_t grid = (total + kBlock - 1U) / kBlock;
    pack_rowmajor_to_colmajor_half_kernel<<<grid, kBlock>>>(d_src_rowmajor, d_dst_colmajor, rows, dim);
    return launch_ok();
}

bool launch_row_norms_f32(
    const float* d_matrix_rowmajor,
    float* d_norms,
    std::uint32_t rows,
    std::size_t dim) {
    constexpr std::uint32_t kBlock = 128U;
    const std::uint32_t grid = (rows + kBlock - 1U) / kBlock;
    row_norms_f32_kernel<<<grid, kBlock>>>(d_matrix_rowmajor, d_norms, rows, dim);
    return launch_ok();
}

bool launch_argmin_from_dot_kn(
    const float* d_dot_kn,
    const float* d_vector_norms,
    const float* d_centroid_norms,
    std::uint32_t* d_assignments,
    float* d_min_dists,
    std::uint32_t n,
    std::uint32_t k) {
    constexpr std::uint32_t kBlock = 128U;
    const std::uint32_t grid = (n + kBlock - 1U) / kBlock;
    argmin_from_dot_kn_kernel<<<grid, kBlock>>>(
        d_dot_kn,
        d_vector_norms,
        d_centroid_norms,
        d_assignments,
        d_min_dists,
        n,
        k);
    return launch_ok();
}

}  // namespace vector_db_v3::kmeans::cuda
