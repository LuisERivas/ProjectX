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

enum class PrecisionPreference {
    Auto = 0,
    FP32 = 1,
    TensorFP16 = 2,
};

struct RuntimeInfo {
    bool observed = false;
    bool cuda_compiled = false;
    bool cuda_available = false;
    bool tensor_compiled = false;
    bool tensor_available = false;
    bool tensor_effective = false;
    bool tensor_active = false;
    std::string backend_path = "cpu";
    std::string gpu_arch_class = "unknown";
    std::string fallback_reason;
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
bool tensor_backend_compiled();
bool tensor_backend_available(std::string* reason);
RuntimeInfo last_runtime_info();
void set_runtime_info_for_stage(const RuntimeInfo& info);

}  // namespace vector_db_v3::kmeans
