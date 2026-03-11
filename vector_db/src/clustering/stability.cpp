#include "vector_db/clustering.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vector_db {

namespace {
constexpr std::size_t kDim = 1024;

double entropy(const std::unordered_map<std::uint32_t, std::size_t>& counts, std::size_t n) {
    if (n == 0) {
        return 0.0;
    }
    double h = 0.0;
    for (const auto& kv : counts) {
        const double p = static_cast<double>(kv.second) / static_cast<double>(n);
        if (p > 0.0) {
            h -= p * std::log(p);
        }
    }
    return h;
}

double nmi(const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }
    using Pair = std::uint64_t;
    std::unordered_map<std::uint32_t, std::size_t> ca;
    std::unordered_map<std::uint32_t, std::size_t> cb;
    std::unordered_map<Pair, std::size_t> cab;
    for (std::size_t i = 0; i < a.size(); ++i) {
        ca[a[i]] += 1;
        cb[b[i]] += 1;
        const Pair key = (static_cast<Pair>(a[i]) << 32U) | static_cast<Pair>(b[i]);
        cab[key] += 1;
    }
    const double n = static_cast<double>(a.size());
    double mi = 0.0;
    for (const auto& kv : cab) {
        const std::uint32_t la = static_cast<std::uint32_t>(kv.first >> 32U);
        const std::uint32_t lb = static_cast<std::uint32_t>(kv.first & 0xFFFFFFFFULL);
        const double pxy = static_cast<double>(kv.second) / n;
        const double px = static_cast<double>(ca[la]) / n;
        const double py = static_cast<double>(cb[lb]) / n;
        if (pxy > 0.0 && px > 0.0 && py > 0.0) {
            mi += pxy * std::log(pxy / (px * py));
        }
    }
    const double ha = entropy(ca, a.size());
    const double hb = entropy(cb, b.size());
    if (ha <= 0.0 || hb <= 0.0) {
        return 0.0;
    }
    return mi / std::sqrt(ha * hb);
}

double jaccard(
    const std::vector<std::vector<std::uint32_t>>& a,
    const std::vector<std::vector<std::uint32_t>>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::unordered_set<std::uint32_t> sa(a[i].begin(), a[i].end());
        std::unordered_set<std::uint32_t> sb(b[i].begin(), b[i].end());
        std::size_t inter = 0;
        for (std::uint32_t x : sa) {
            if (sb.find(x) != sb.end()) {
                inter += 1;
            }
        }
        const std::size_t uni = sa.size() + sb.size() - inter;
        total += (uni == 0) ? 1.0 : (static_cast<double>(inter) / static_cast<double>(uni));
    }
    return total / static_cast<double>(a.size());
}

double centroid_drift(const KMeansModel& a, const KMeansModel& b) {
    if (a.k != b.k || a.k == 0) {
        return 1.0;
    }
    std::vector<bool> used(b.k, false);
    double sum = 0.0;
    for (std::size_t i = 0; i < a.k; ++i) {
        double best = 1.0;
        std::size_t best_j = 0;
        for (std::size_t j = 0; j < b.k; ++j) {
            if (used[j]) {
                continue;
            }
            double dot = 0.0;
            for (std::size_t d = 0; d < 1024; ++d) {
                dot += static_cast<double>(a.centroids[i * 1024 + d]) * static_cast<double>(b.centroids[j * 1024 + d]);
            }
            const double dist = 1.0 - dot;
            if (dist < best) {
                best = dist;
                best_j = j;
            }
        }
        used[best_j] = true;
        sum += best;
    }
    return sum / static_cast<double>(a.k);
}

}  // namespace

Status evaluate_stability(
    const std::vector<std::vector<float>>& vectors,
    std::size_t chosen_k,
    const InitialClusteringConfig& cfg,
    StabilityMetrics* out_metrics) {
    if (out_metrics == nullptr) {
        return Status::Error("stability output pointer is null");
    }
    if (cfg.stability_runs < 2) {
        return Status::Error("stability requires at least 2 runs");
    }
    std::vector<float> vectors_row_major;
    vectors_row_major.reserve(vectors.size() * kDim);
    for (const auto& v : vectors) {
        vectors_row_major.insert(vectors_row_major.end(), v.begin(), v.end());
    }
    std::vector<KMeansModel> models;
    models.reserve(cfg.stability_runs);
    for (std::size_t r = 0; r < cfg.stability_runs; ++r) {
        KMeansModel model;
        const Status s = fit_spherical_kmeans_packed(
            vectors,
            vectors_row_major,
            chosen_k,
            cfg,
            cfg.seed + 1000U + static_cast<std::uint32_t>(r),
            &model);
        if (!s.ok) {
            return s;
        }
        models.push_back(std::move(model));
    }

    std::vector<double> nmis;
    std::vector<double> jaccs;
    std::vector<double> drifts;
    for (std::size_t i = 0; i < models.size(); ++i) {
        for (std::size_t j = i + 1; j < models.size(); ++j) {
            nmis.push_back(nmi(models[i].labels, models[j].labels));
            jaccs.push_back(jaccard(models[i].assignments, models[j].assignments));
            drifts.push_back(centroid_drift(models[i], models[j]));
        }
    }

    auto mean = [](const std::vector<double>& v) -> double {
        if (v.empty()) {
            return 0.0;
        }
        const double s = std::accumulate(v.begin(), v.end(), 0.0);
        return s / static_cast<double>(v.size());
    };
    const double mean_nmi = mean(nmis);
    const double mean_jacc = mean(jaccs);
    const double mean_drift = mean(drifts);
    double var_nmi = 0.0;
    for (double x : nmis) {
        const double d = x - mean_nmi;
        var_nmi += d * d;
    }
    if (!nmis.empty()) {
        var_nmi /= static_cast<double>(nmis.size());
    }
    const double std_nmi = std::sqrt(var_nmi);

    out_metrics->mean_nmi = mean_nmi;
    out_metrics->std_nmi = std_nmi;
    out_metrics->mean_jaccard = mean_jacc;
    out_metrics->mean_centroid_drift = mean_drift;
    out_metrics->passed =
        (mean_nmi >= 0.90) &&
        (std_nmi <= 0.03) &&
        (mean_jacc >= 0.85) &&
        (mean_drift <= 0.02);
    return Status::Ok();
}

}  // namespace vector_db

