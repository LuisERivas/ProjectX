#include "vector_db_v3/kmeans_backend.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace vector_db_v3::kmeans {

Status run_kmeans_cuda(
    const std::vector<std::vector<float>>& vectors,
    std::uint32_t k,
    std::uint32_t max_iterations,
    KMeansResult* out);

namespace {

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
        return run_kmeans_cpu(vectors, k, max_iterations, out);
    }

    if (preference == BackendPreference::Cuda) {
        const Status cuda_st = run_kmeans_cuda(vectors, k, max_iterations, out);
        if (cuda_st.ok) {
            if (backend_used != nullptr) {
                *backend_used = "cuda";
            }
            return cuda_st;
        }
        return cuda_st;
    }

    // Auto: try CUDA first, fallback to CPU.
    const Status cuda_st = run_kmeans_cuda(vectors, k, max_iterations, out);
    if (cuda_st.ok) {
        if (backend_used != nullptr) {
            *backend_used = "cuda";
        }
        return cuda_st;
    }
    if (backend_used != nullptr) {
        *backend_used = "cpu";
    }
    return run_kmeans_cpu(vectors, k, max_iterations, out);
}

}  // namespace vector_db_v3::kmeans
