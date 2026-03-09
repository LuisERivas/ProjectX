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

float dot_raw(const float* a, const float* b, std::size_t dim) {
    float s = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

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

void select_top_m(const std::vector<float>& scores, std::size_t m, std::vector<std::uint32_t>* out) {
    std::vector<std::pair<float, std::uint32_t>> pairs;
    pairs.reserve(scores.size());
    for (std::size_t i = 0; i < scores.size(); ++i) {
        pairs.push_back({scores[i], static_cast<std::uint32_t>(i)});
    }
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    out->clear();
    const std::size_t take = std::min(m, pairs.size());
    for (std::size_t i = 0; i < take; ++i) {
        out->push_back(pairs[i].second);
    }
}

Status compute_scores(
    const std::vector<std::vector<float>>& vectors,
    const std::vector<float>& centroids,
    std::size_t k,
    std::vector<float>* out_scores,
    bool* used_cuda) {
    const std::size_t n = vectors.size();
    out_scores->assign(n * k, 0.0f);
    *used_cuda = false;

    std::vector<float> vectors_row_major;
    vectors_row_major.reserve(n * kDim);
    for (const auto& v : vectors) {
        vectors_row_major.insert(vectors_row_major.end(), v.begin(), v.end());
    }

    if (cuda_dot_products_available()) {
        if (const Status s = cuda_compute_dot_products(vectors_row_major, centroids, n, k, kDim, out_scores); s.ok) {
            *used_cuda = true;
            return Status::Ok();
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t c = 0; c < k; ++c) {
            const float* x = vectors[i].data();
            const float* mu = &centroids[c * kDim];
            (*out_scores)[i * k + c] = dot_raw(x, mu, kDim);
        }
    }
    return Status::Ok();
}

Status run_kmeans_impl(
    const std::vector<std::vector<float>>& vectors,
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

    std::vector<float> scores;
    bool used_cuda_once = false;
    for (std::size_t iter = 0; iter < cfg.iters; ++iter) {
        bool used_cuda_iter = false;
        if (const Status s = compute_scores(vectors, centroids, k, &scores, &used_cuda_iter); !s.ok) {
            return s;
        }
        used_cuda_once = used_cuda_once || used_cuda_iter;

        std::vector<float> sums(k * kDim, 0.0f);
        std::vector<std::size_t> counts(k, 0);
        for (std::size_t i = 0; i < vectors.size(); ++i) {
            std::size_t best = 0;
            float best_score = -std::numeric_limits<float>::infinity();
            for (std::size_t c = 0; c < k; ++c) {
                const float s = scores[i * k + c];
                if (s > best_score) {
                    best_score = s;
                    best = c;
                }
            }
            counts[best] += 1;
            for (std::size_t d = 0; d < kDim; ++d) {
                sums[best * kDim + d] += vectors[i][d];
            }
        }

        for (std::size_t c = 0; c < k; ++c) {
            if (counts[c] == 0) {
                const auto& src = vectors[idx[(c + iter) % vectors.size()]];
                std::copy(src.begin(), src.end(), centroids.begin() + static_cast<std::ptrdiff_t>(c * kDim));
                continue;
            }
            float norm_sq = 0.0f;
            for (std::size_t d = 0; d < kDim; ++d) {
                const float v = sums[c * kDim + d] / static_cast<float>(counts[c]);
                centroids[c * kDim + d] = v;
                norm_sq += v * v;
            }
            const float norm = std::sqrt(std::max(1e-12f, norm_sq));
            for (std::size_t d = 0; d < kDim; ++d) {
                centroids[c * kDim + d] /= norm;
            }
        }
    }

    bool used_cuda_final = false;
    if (const Status s = compute_scores(vectors, centroids, k, &scores, &used_cuda_final); !s.ok) {
        return s;
    }
    used_cuda_once = used_cuda_once || used_cuda_final;

    std::vector<std::vector<std::uint32_t>> top_assign(vectors.size());
    std::vector<std::uint32_t> labels(vectors.size(), 0);
    double objective = 0.0;
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        std::vector<float> row(k, 0.0f);
        for (std::size_t c = 0; c < k; ++c) {
            row[c] = scores[i * k + c];
        }
        select_top_m(row, cfg.top_m, &top_assign[i]);
        labels[i] = top_assign[i].empty() ? 0U : top_assign[i][0];
        const float best = top_assign[i].empty() ? -1.0f : row[top_assign[i][0]];
        objective += (1.0 - static_cast<double>(best));
    }
    objective /= static_cast<double>(vectors.size());

    out_model->k = k;
    out_model->objective = objective;
    out_model->used_cuda = used_cuda_once;
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
    return run_kmeans_impl(vectors, k, cfg, seed, out_model);
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

    std::unordered_map<std::size_t, KMeansModel> model_cache;
    auto ensure_model = [&](std::size_t k) -> Status {
        if (model_cache.find(k) != model_cache.end()) {
            return Status::Ok();
        }
        KMeansModel m;
        const Status s = run_kmeans_impl(vectors, k, cfg, cfg.seed + static_cast<std::uint32_t>(k), &m);
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
Status cuda_compute_dot_products(
    const std::vector<float>&,
    const std::vector<float>&,
    std::size_t,
    std::size_t,
    std::size_t,
    std::vector<float>*) {
    return Status::Error("cuda support is not compiled");
}
#endif

}  // namespace vector_db

