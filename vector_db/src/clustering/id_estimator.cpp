#include "vector_db/clustering.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace vector_db {

namespace {

double dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return s;
}

std::size_t clamp_sample_size(std::size_t n, std::size_t min_sample, std::size_t max_sample) {
    const std::size_t bounded_min = std::max<std::size_t>(8, min_sample);
    const std::size_t bounded_max = std::max(bounded_min, max_sample);
    return std::min(bounded_max, std::max(bounded_min, n));
}

std::size_t round_down_pow2(std::size_t v) {
    std::size_t p = 1;
    while ((p << 1U) <= v) {
        p <<= 1U;
    }
    return p;
}

std::size_t round_up_pow2(std::size_t v) {
    std::size_t p = 1;
    while (p < v) {
        p <<= 1U;
    }
    return p;
}

}  // namespace

Status estimate_intrinsic_dimensionality(
    const std::vector<std::vector<float>>& vectors,
    std::uint32_t seed,
    std::size_t min_sample,
    std::size_t max_sample,
    IdEstimateRange* out) {
    if (out == nullptr) {
        return Status::Error("id estimate output pointer is null");
    }
    if (vectors.empty()) {
        return Status::Error("cannot estimate intrinsic dimensionality on empty vectors");
    }

    const std::size_t sample_size = clamp_sample_size(vectors.size(), min_sample, max_sample);
    std::vector<std::size_t> all_idx(vectors.size(), 0);
    std::iota(all_idx.begin(), all_idx.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(all_idx.begin(), all_idx.end(), rng);
    all_idx.resize(sample_size);

    std::vector<double> local_dims;
    local_dims.reserve(sample_size);
    for (std::size_t i = 0; i < all_idx.size(); ++i) {
        const auto& xi = vectors[all_idx[i]];
        double best = std::numeric_limits<double>::infinity();
        double second = std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < all_idx.size(); ++j) {
            if (i == j) {
                continue;
            }
            const auto& xj = vectors[all_idx[j]];
            const double cosine = std::max(-1.0, std::min(1.0, dot_product(xi, xj)));
            const double dist = std::max(1e-9, 1.0 - cosine);
            if (dist < best) {
                second = best;
                best = dist;
            } else if (dist < second) {
                second = dist;
            }
        }
        if (!std::isfinite(best) || !std::isfinite(second) || second <= best || best <= 0.0) {
            continue;
        }
        const double m = std::log(2.0) / std::log(second / best);
        if (std::isfinite(m) && m > 0.0) {
            local_dims.push_back(m);
        }
    }
    if (local_dims.empty()) {
        return Status::Error("failed to compute local intrinsic dimensionality");
    }

    std::sort(local_dims.begin(), local_dims.end());
    const std::size_t p25 = static_cast<std::size_t>(0.25 * static_cast<double>(local_dims.size() - 1));
    const std::size_t p75 = static_cast<std::size_t>(0.75 * static_cast<double>(local_dims.size() - 1));
    const double m_low = std::max(2.0, std::min(512.0, local_dims[p25]));
    const double m_high = std::max(m_low, std::min(512.0, local_dims[p75]));
    const double m_mid = 0.5 * (m_low + m_high);

    const double n = static_cast<double>(vectors.size());
    const double k_center = std::sqrt(n) * std::sqrt(m_mid / 32.0);
    const std::size_t k_min_raw = static_cast<std::size_t>(std::max(1.0, 0.5 * k_center));
    const std::size_t k_max_raw = static_cast<std::size_t>(std::max(1.0, 2.0 * k_center));

    std::size_t k_min = std::max<std::size_t>(8, round_down_pow2(k_min_raw));
    std::size_t k_max = std::min<std::size_t>(8192, round_up_pow2(k_max_raw));
    if (k_min >= k_max) {
        k_min = std::max<std::size_t>(8, k_max / 2);
    }

    out->sample_size = sample_size;
    out->m_low = m_low;
    out->m_high = m_high;
    out->k_min = k_min;
    out->k_max = k_max;
    return Status::Ok();
}

}  // namespace vector_db

