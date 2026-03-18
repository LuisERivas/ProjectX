#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>

namespace vector_db_v3::kmeans::cuda {

bool launch_pack_rowmajor_to_colmajor_half(
    const float* d_src_rowmajor,
    __half* d_dst_colmajor,
    std::uint32_t rows,
    std::size_t dim);

bool launch_row_norms_f32(
    const float* d_matrix_rowmajor,
    float* d_norms,
    std::uint32_t rows,
    std::size_t dim);

bool launch_argmin_from_dot_kn(
    const float* d_dot_kn,
    const float* d_vector_norms,
    const float* d_centroid_norms,
    std::uint32_t* d_assignments,
    float* d_min_dists,
    std::uint32_t n,
    std::uint32_t k);

}  // namespace vector_db_v3::kmeans::cuda
