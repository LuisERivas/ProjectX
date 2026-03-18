#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vector_db_v3/kmeans_backend.hpp"

namespace vector_db_v3::kmeans {

bool tensor_path_effective(std::uint32_t vector_count, std::uint32_t k, std::size_t dim, std::string* reason);

Status run_kmeans_cuda_tensor(
    const std::vector<std::vector<float>>& vectors,
    std::uint32_t k,
    std::uint32_t max_iterations,
    KMeansResult* out);

}  // namespace vector_db_v3::kmeans
