#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vector_db_v3/status.hpp"
#include "vector_db_v3/vector_store.hpp"

namespace vector_db_v3::kmeans {

enum class BackendPreference {
    Auto = 0,
    Cpu = 1,
    Cuda = 2,
};

struct KMeansResult {
    std::vector<std::vector<float>> centroids;
    std::vector<std::uint32_t> assignments;
    double objective = 0.0;
};

Status run_kmeans(
    const std::vector<std::vector<float>>& vectors,
    std::uint32_t k,
    std::uint32_t max_iterations,
    BackendPreference preference,
    KMeansResult* out,
    std::string* backend_used);

bool cuda_backend_compiled();
bool cuda_backend_available(std::string* reason);

}  // namespace vector_db_v3::kmeans
