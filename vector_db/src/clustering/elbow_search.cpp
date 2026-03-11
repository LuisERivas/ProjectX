#include "vector_db/clustering.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <unordered_map>

namespace vector_db {

namespace {

constexpr std::size_t kDim = 1024;

std::vector<std::size_t> power_of_two_grid(std::size_t k_min, std::size_t k_max) {
    std::vector<std::size_t> out;
    if (k_min == 0 || k_max == 0 || k_min > k_max) {
        return out;
    }
    std::size_t k = k_min;
    out.push_back(k);
    while (k < k_max) {
        if (k > (std::numeric_limits<std::size_t>::max() >> 1U)) {
            break;
        }
        k <<= 1U;
        out.push_back(k);
    }
    return out;
}

bool is_in_norm_guard(const std::vector<float>& x, double min_norm, double max_norm) {
    double s = 0.0;
    for (float v : x) {
        s += static_cast<double>(v) * static_cast<double>(v);
    }
    const double n = std::sqrt(s);
    return n >= min_norm && n <= max_norm;
}

std::vector<float> pack_vectors_row_major(const std::vector<std::vector<float>>& vectors) {
    std::vector<float> out;
    out.reserve(vectors.size() * kDim);
    for (const auto& v : vectors) {
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

Status run_kmeans_impl(
    const std::vector<std::vector<float>>& vectors,
    const std::vector<float>& vectors_row_major,
    std::size_t k,
    const InitialClusteringConfig& cfg,
    std::uint32_t seed,
    KMeansModel* out_model) {
    if (vectors.empty()) {
        return Status::Error("kmeans requires non-empty vectors");
    }
    if (k == 0) {
        return Status::Error("kmeans requires k > 0");
    }
    if (k > vectors.size()) {
        return Status::Error("k cannot exceed vector count");
    }
    if (!cuda_dot_products_available() || !cuda_assignment_kernels_available()) {
        return Status::Error(
            "GPU-only clustering requires CUDA scoring and assignment kernels; "
            "build with CUDA support and a compatible GPU");
    }
    if (vectors_row_major.size() != vectors.size() * kDim) {
        return Status::Error("GPU-only clustering requires a contiguous packed vectors buffer");
    }
    for (const auto& v : vectors) {
        if (v.size() != kDim) {
            return Status::Error("unexpected vector dimension for kmeans");
        }
        if (!is_in_norm_guard(v, cfg.min_norm_guard, cfg.max_norm_guard)) {
            return Status::Error("vector norm guard failed; expected pre-normalized vectors");
        }
    }

    std::mt19937 rng(seed);
    std::vector<std::size_t> idx(vectors.size(), 0);
    for (std::size_t i = 0; i < idx.size(); ++i) {
        idx[i] = i;
    }
    std::shuffle(idx.begin(), idx.end(), rng);

    std::vector<float> centroids(k * kDim, 0.0f);
    for (std::size_t c = 0; c < k; ++c) {
        const auto& src = vectors[idx[c % vectors.size()]];
        std::copy(src.begin(), src.end(), centroids.begin() + static_cast<std::ptrdiff_t>(c * kDim));
    }

    bool tensor_core_once = false;
    std::string backend_used = "cuda";
    double scoring_ms_total = 0.0;
    std::size_t scoring_calls = 0;
    for (std::size_t iter = 0; iter < cfg.iters; ++iter) {
        bool tensor_iter = false;
        std::string backend_iter = "cuda";
        double scoring_ms = 0.0;
        std::vector<std::uint32_t> labels_gpu;
        std::vector<float> best_scores;
        std::vector<float> centroids_gpu;
        if (const Status s = cuda_kmeans_iteration_top1(
                vectors_row_major,
                centroids,
                vectors.size(),
                k,
                kDim,
                &centroids_gpu,
                &labels_gpu,
                &best_scores,
                &tensor_iter,
                &backend_iter,
                &scoring_ms);
            !s.ok) {
            return Status::Error("GPU kmeans iteration failed: " + s.message);
        }
        centroids = std::move(centroids_gpu);
        tensor_core_once = tensor_core_once || tensor_iter;
        backend_used = backend_iter;
        scoring_ms_total += scoring_ms;
        scoring_calls += 1;
    }

    bool tensor_final = false;
    std::string backend_final = "cuda";
    double scoring_ms_final = 0.0;
    std::vector<std::vector<std::uint32_t>> top_assign(vectors.size());
    std::vector<float> scores;
    if (const Status s = cuda_topm_from_centroids(
            vectors_row_major,
            centroids,
            vectors.size(),
            k,
            kDim,
            cfg.top_m,
            &top_assign,
            &scores,
            &tensor_final,
            &backend_final,
            &scoring_ms_final);
        !s.ok) {
        return Status::Error("GPU top-m assignment failed: " + s.message);
    }
    tensor_core_once = tensor_core_once || tensor_final;
    backend_used = backend_final;
    scoring_ms_total += scoring_ms_final;
    scoring_calls += 1;

    std::vector<std::uint32_t> labels(vectors.size(), 0);
    double objective = 0.0;
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        labels[i] = top_assign[i].empty() ? 0U : top_assign[i][0];
        const float best = top_assign[i].empty() ? -1.0f : scores[i * k + top_assign[i][0]];
        objective += (1.0 - static_cast<double>(best));
    }
    objective /= static_cast<double>(vectors.size());

    out_model->k = k;
    out_model->objective = objective;
    out_model->used_cuda = true;
    out_model->tensor_core_enabled = tensor_core_once;
    out_model->gpu_backend = backend_used;
    out_model->scoring_ms_total = scoring_ms_total;
    out_model->scoring_calls = scoring_calls;
    out_model->centroids = std::move(centroids);
    out_model->assignments = std::move(top_assign);
    out_model->labels = std::move(labels);
    return Status::Ok();
}

}  // namespace

Status fit_spherical_kmeans(
    const std::vector<std::vector<float>>& vectors,
    std::size_t k,
    const InitialClusteringConfig& cfg,
    std::uint32_t seed,
    KMeansModel* out_model) {
    if (out_model == nullptr) {
        return Status::Error("null output model for fit_spherical_kmeans");
    }
    const std::vector<float> vectors_row_major = pack_vectors_row_major(vectors);
    return run_kmeans_impl(vectors, vectors_row_major, k, cfg, seed, out_model);
}

Status fit_spherical_kmeans_packed(
    const std::vector<std::vector<float>>& vectors,
    const std::vector<float>& vectors_row_major,
    std::size_t k,
    const InitialClusteringConfig& cfg,
    std::uint32_t seed,
    KMeansModel* out_model) {
    if (out_model == nullptr) {
        return Status::Error("null output model for fit_spherical_kmeans_packed");
    }
    return run_kmeans_impl(vectors, vectors_row_major, k, cfg, seed, out_model);
}

Status select_k_binary_elbow(
    const std::vector<std::vector<float>>& vectors,
    const IdEstimateRange& id_range,
    const InitialClusteringConfig& cfg,
    KMeansModel* out_best_model,
    ElbowSelection* out_selection) {
    if (out_best_model == nullptr || out_selection == nullptr) {
        return Status::Error("binary elbow output pointers must be non-null");
    }
    const auto grid = power_of_two_grid(id_range.k_min, id_range.k_max);
    if (grid.size() < 2) {
        return Status::Error("invalid k grid for binary elbow selection");
    }
    const std::vector<float> vectors_row_major = pack_vectors_row_major(vectors);

    std::unordered_map<std::size_t, KMeansModel> model_cache;
    auto ensure_model = [&](std::size_t k) -> Status {
        if (model_cache.find(k) != model_cache.end()) {
            return Status::Ok();
        }
        KMeansModel m;
        const Status s = run_kmeans_impl(
            vectors,
            vectors_row_major,
            k,
            cfg,
            cfg.seed + static_cast<std::uint32_t>(k),
            &m);
        if (!s.ok) {
            return s;
        }
        model_cache.emplace(k, std::move(m));
        return Status::Ok();
    };

    std::size_t lo = 0;
    std::size_t hi = grid.size() - 2;
    std::size_t best_idx = grid.size() - 1;
    bool found = false;
    std::unordered_map<std::size_t, double> gain;

    while (lo <= hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        const std::size_t k = grid[mid];
        const std::size_t k2 = grid[mid + 1];
        if (const Status s = ensure_model(k); !s.ok) {
            return s;
        }
        if (const Status s = ensure_model(k2); !s.ok) {
            return s;
        }
        const double j1 = model_cache[k].objective;
        const double j2 = model_cache[k2].objective;
        const double g = (j1 <= 0.0) ? 0.0 : ((j1 - j2) / j1);
        gain[k] = g;
        if (g <= cfg.elbow_gain_threshold) {
            found = true;
            best_idx = mid;
            if (mid == 0) {
                break;
            }
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    std::size_t chosen_k = found ? grid[best_idx] : grid.back();
    bool fallback = !found;
    if (found && best_idx > 0) {
        const std::size_t prev_k = grid[best_idx - 1];
        if (gain.find(prev_k) == gain.end()) {
            const std::size_t prev_k2 = grid[best_idx];
            if (const Status s = ensure_model(prev_k); !s.ok) {
                return s;
            }
            const double j1 = model_cache[prev_k].objective;
            const double j2 = model_cache[prev_k2].objective;
            gain[prev_k] = (j1 <= 0.0) ? 0.0 : ((j1 - j2) / j1);
        }
        const double flatness = std::abs(gain[prev_k] - gain[chosen_k]);
        if (flatness > cfg.elbow_flat_threshold) {
            chosen_k = grid.back();
            fallback = true;
        }
    }

    if (const Status s = ensure_model(chosen_k); !s.ok) {
        return s;
    }
    *out_best_model = model_cache[chosen_k];
    out_selection->chosen_k = chosen_k;
    out_selection->used_fallback = fallback;
    out_selection->trace.clear();
    for (std::size_t i = 0; i + 1 < grid.size(); ++i) {
        const std::size_t k = grid[i];
        const std::size_t k2 = grid[i + 1];
        if (const Status s = ensure_model(k); !s.ok) {
            return s;
        }
        if (const Status s = ensure_model(k2); !s.ok) {
            return s;
        }
        const double g = (model_cache[k].objective <= 0.0)
            ? 0.0
            : ((model_cache[k].objective - model_cache[k2].objective) / model_cache[k].objective);
        out_selection->trace.push_back(ElbowPoint{k, model_cache[k].objective, g});
    }
    if (const Status s = ensure_model(grid.back()); !s.ok) {
        return s;
    }
    out_selection->trace.push_back(ElbowPoint{grid.back(), model_cache[grid.back()].objective, 0.0});
    return Status::Ok();
}

#ifndef VECTOR_DB_USE_CUDA
bool cuda_dot_products_available() { return false; }
bool cuda_assignment_kernels_available() { return false; }
Status cuda_compute_dot_products(
    const std::vector<float>&,
    const std::vector<float>&,
    std::size_t,
    std::size_t,
    std::size_t,
    std::vector<float>*,
    bool* out_tensor_core_enabled,
    std::string* out_backend_name) {
    if (out_tensor_core_enabled != nullptr) {
        *out_tensor_core_enabled = false;
    }
    if (out_backend_name != nullptr) {
        *out_backend_name = "cpu";
    }
    return Status::Error("cuda support is not compiled");
}
Status cuda_assign_top1_labels(
    const std::vector<float>&,
    std::size_t,
    std::size_t,
    std::vector<std::uint32_t>*,
    std::vector<float>*) {
    return Status::Error("cuda support is not compiled");
}
Status cuda_reduce_centroids_top1(
    const std::vector<float>&,
    const std::vector<std::uint32_t>&,
    std::size_t,
    std::size_t,
    std::size_t,
    std::vector<float>*) {
    return Status::Error("cuda support is not compiled");
}
Status cuda_kmeans_iteration_top1(
    const std::vector<float>&,
    const std::vector<float>&,
    std::size_t,
    std::size_t,
    std::size_t,
    std::vector<float>*,
    std::vector<std::uint32_t>*,
    std::vector<float>*,
    bool* out_tensor_core_enabled,
    std::string* out_backend_name,
    double* out_scoring_ms) {
    if (out_tensor_core_enabled != nullptr) {
        *out_tensor_core_enabled = false;
    }
    if (out_backend_name != nullptr) {
        *out_backend_name = "cpu";
    }
    if (out_scoring_ms != nullptr) {
        *out_scoring_ms = 0.0;
    }
    return Status::Error("cuda support is not compiled");
}
Status cuda_topm_from_centroids(
    const std::vector<float>&,
    const std::vector<float>&,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::vector<std::vector<std::uint32_t>>*,
    std::vector<float>*,
    bool* out_tensor_core_enabled,
    std::string* out_backend_name,
    double* out_scoring_ms) {
    if (out_tensor_core_enabled != nullptr) {
        *out_tensor_core_enabled = false;
    }
    if (out_backend_name != nullptr) {
        *out_backend_name = "cpu";
    }
    if (out_scoring_ms != nullptr) {
        *out_scoring_ms = 0.0;
    }
    return Status::Error("cuda support is not compiled");
}
#endif

}  // namespace vector_db

