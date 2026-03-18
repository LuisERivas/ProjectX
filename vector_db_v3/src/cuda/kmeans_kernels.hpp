#pragma once

#include <cstddef>
#include <cstdint>

namespace vector_db_v3::kmeans::cuda {

bool launch_assignment_kernel(
    const float* d_vectors,
    const float* d_centroids,
    std::uint32_t* d_assignments,
    float* d_min_dists,
    std::uint32_t vector_count,
    std::uint32_t k,
    std::size_t dim);

bool launch_accumulate_kernel(
    const float* d_vectors,
    const std::uint32_t* d_assignments,
    float* d_sums,
    std::uint32_t* d_counts,
    std::uint32_t vector_count,
    std::uint32_t k,
    std::size_t dim);

bool launch_update_kernel(
    float* d_centroids,
    const float* d_sums,
    const std::uint32_t* d_counts,
    std::uint32_t k,
    std::size_t dim);

bool launch_objective_kernel(
    const float* d_vectors,
    const float* d_centroids,
    const std::uint32_t* d_assignments,
    float* d_objective,
    std::uint32_t vector_count,
    std::size_t dim);

}  // namespace vector_db_v3::kmeans::cuda
