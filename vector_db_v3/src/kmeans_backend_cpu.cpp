#include "vector_db_v3/kmeans_backend.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <string>

namespace vector_db_v3::kmeans {

Status run_kmeans_cuda(
    const std::vector<std::vector<float>>& vectors,
    std::uint32_t k,
    std::uint32_t max_iterations,
    PrecisionPreference precision_preference,
    bool tensor_required,
    KMeansResult* out);

namespace {

RuntimeInfo g_last_runtime_info{};

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

PrecisionPreference parse_precision_preference() {
    const char* value = std::getenv("VECTOR_DB_V3_KMEANS_PRECISION");
    if (value == nullptr || *value == '\0') {
        return PrecisionPreference::Auto;
    }
    const std::string pref = lowercase(std::string(value));
    if (pref == "fp32" || pref == "cuda_fp32") {
        return PrecisionPreference::FP32;
    }
    if (pref == "tensor" || pref == "fp16" || pref == "tensor_fp16") {
        return PrecisionPreference::TensorFP16;
    }
    return PrecisionPreference::Auto;
}

bool env_truthy(const char* value) {
    if (value == nullptr) {
        return false;
    }
    const std::string v = lowercase(std::string(value));
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

bool tensor_required_by_policy() {
    const char* policy = std::getenv("VECTOR_DB_V3_TENSOR_POLICY");
    if (policy == nullptr || *policy == '\0') {
        return false;
    }
    return lowercase(std::string(policy)) == "required";
}

double squared_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double out = 0.0;
    for (std::size_t i = 0; i < kVectorDim; ++i) {
        const double delta = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        out += delta * delta;
    }
    return out;
}

Status run_kmeans_cpu(
    const std::vector<std::vector<float>>& vectors,
    std::uint32_t k,
    std::uint32_t max_iterations,
    KMeansResult* out) {
    if (out == nullptr) {
        return Status::Error("kmeans cpu: out is null");
    }
    *out = KMeansResult{};
    if (vectors.empty() || k == 0U) {
        return Status::Ok();
    }
    k = std::min<std::uint32_t>(k, static_cast<std::uint32_t>(vectors.size()));
    out->centroids.assign(k, std::vector<float>(kVectorDim, 0.0f));
    out->assignments.assign(vectors.size(), 0U);

    for (std::uint32_t c = 0; c < k; ++c) {
        const std::size_t idx = (static_cast<std::size_t>(c) * vectors.size()) / k;
        out->centroids[c] = vectors[idx];
    }

    for (std::uint32_t iter = 0; iter < std::max<std::uint32_t>(1U, max_iterations); ++iter) {
        bool changed = false;
        std::vector<double> min_dist(vectors.size(), std::numeric_limits<double>::infinity());
        for (std::size_t i = 0; i < vectors.size(); ++i) {
            std::uint32_t best = 0U;
            double best_dist = std::numeric_limits<double>::infinity();
            for (std::uint32_t c = 0; c < k; ++c) {
                const double dist = squared_l2(vectors[i], out->centroids[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            min_dist[i] = best_dist;
            if (iter == 0U || out->assignments[i] != best) {
                changed = true;
                out->assignments[i] = best;
            }
        }

        std::vector<std::vector<double>> sums(k, std::vector<double>(kVectorDim, 0.0));
        std::vector<std::uint32_t> counts(k, 0U);
        for (std::size_t i = 0; i < vectors.size(); ++i) {
            const std::uint32_t bucket = out->assignments[i];
            ++counts[bucket];
            for (std::size_t d = 0; d < kVectorDim; ++d) {
                sums[bucket][d] += static_cast<double>(vectors[i][d]);
            }
        }
        for (std::uint32_t c = 0; c < k; ++c) {
            if (counts[c] == 0U) {
                std::size_t worst_idx = 0U;
                double worst_dist = -1.0;
                for (std::size_t i = 0; i < min_dist.size(); ++i) {
                    if (min_dist[i] > worst_dist) {
                        worst_dist = min_dist[i];
                        worst_idx = i;
                    }
                }
                out->assignments[worst_idx] = c;
                counts[c] = 1U;
                for (std::size_t d = 0; d < kVectorDim; ++d) {
                    sums[c][d] = static_cast<double>(vectors[worst_idx][d]);
                }
            }
            for (std::size_t d = 0; d < kVectorDim; ++d) {
                out->centroids[c][d] = static_cast<float>(sums[c][d] / static_cast<double>(counts[c]));
            }
        }
        if (!changed) {
            break;
        }
    }

    out->objective = 0.0;
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        out->objective += squared_l2(vectors[i], out->centroids[out->assignments[i]]);
    }
    return Status::Ok();
}

}  // namespace

Status run_kmeans(
    const std::vector<std::vector<float>>& vectors,
    std::uint32_t k,
    std::uint32_t max_iterations,
    BackendPreference preference,
    KMeansResult* out,
    std::string* backend_used) {
    if (out == nullptr) {
        return Status::Error("kmeans run: out is null");
    }

    if (preference == BackendPreference::Cpu) {
        if (backend_used != nullptr) {
            *backend_used = "cpu";
        }
        RuntimeInfo info{};
        info.backend_path = "cpu";
        set_runtime_info_for_stage(info);
        return run_kmeans_cpu(vectors, k, max_iterations, out);
    }

    const PrecisionPreference precision_pref = parse_precision_preference();
    const bool force_tensor = env_truthy(std::getenv("VECTOR_DB_V3_FORCE_TENSOR_PATH"));
    const bool tensor_required = force_tensor || tensor_required_by_policy();

    if (preference == BackendPreference::Cuda) {
        const Status cuda_st = run_kmeans_cuda(vectors, k, max_iterations, precision_pref, tensor_required, out);
        if (cuda_st.ok) {
            if (backend_used != nullptr) {
                *backend_used = last_runtime_info().backend_path;
            }
            return cuda_st;
        }
        return cuda_st;
    }

    // Auto: try CUDA first, fallback to CPU.
    const Status cuda_st = run_kmeans_cuda(vectors, k, max_iterations, precision_pref, tensor_required, out);
    if (cuda_st.ok) {
        if (backend_used != nullptr) {
            *backend_used = last_runtime_info().backend_path;
        }
        return cuda_st;
    }
    if (backend_used != nullptr) {
        *backend_used = "cpu";
    }
    RuntimeInfo info{};
    info.backend_path = "cpu";
    info.fallback_reason = tensor_required ? "tensor_path_unavailable_or_not_effective" : "cuda_path_unavailable";
    set_runtime_info_for_stage(info);
    return run_kmeans_cpu(vectors, k, max_iterations, out);
}

RuntimeInfo last_runtime_info() {
    return g_last_runtime_info;
}

void set_runtime_info_for_stage(const RuntimeInfo& info) {
    g_last_runtime_info = info;
}

}  // namespace vector_db_v3::kmeans
