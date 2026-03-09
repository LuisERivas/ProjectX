#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vector_db/status.hpp"

namespace vector_db {

struct IdEstimateRange {
    std::size_t sample_size = 0;
    double m_low = 0.0;
    double m_high = 0.0;
    std::size_t k_min = 8;
    std::size_t k_max = 8;
};

struct ElbowPoint {
    std::size_t k = 0;
    double objective = 0.0;
    double gain_to_2k = 0.0;
};

struct ElbowSelection {
    std::size_t chosen_k = 0;
    bool used_fallback = false;
    std::vector<ElbowPoint> trace;
};

struct KMeansModel {
    std::size_t k = 0;
    double objective = 0.0;
    bool used_cuda = false;
    bool tensor_core_enabled = false;
    std::string gpu_backend = "cpu";
    double scoring_ms_total = 0.0;
    std::size_t scoring_calls = 0;
    std::vector<float> centroids;  // row-major [k][1024]
    std::vector<std::vector<std::uint32_t>> assignments;  // top-m centroid ids per vector
    std::vector<std::uint32_t> labels;  // top-1 centroid id per vector
};

struct StabilityMetrics {
    double mean_nmi = 0.0;
    double std_nmi = 0.0;
    double mean_jaccard = 0.0;
    double mean_centroid_drift = 0.0;
    bool passed = false;
};

struct InitialClusteringConfig {
    std::uint32_t seed = 1234;
    std::size_t min_sample = 256;
    std::size_t max_sample = 4096;
    std::size_t minibatch_size = 256;
    std::size_t iters = 20;
    std::size_t top_m = 2;
    std::size_t stability_runs = 5;
    double elbow_gain_threshold = 0.08;
    double elbow_flat_threshold = 0.02;
    double min_norm_guard = 0.999;
    double max_norm_guard = 1.001;
};

Status estimate_intrinsic_dimensionality(
    const std::vector<std::vector<float>>& vectors,
    std::uint32_t seed,
    std::size_t min_sample,
    std::size_t max_sample,
    IdEstimateRange* out);

Status fit_spherical_kmeans(
    const std::vector<std::vector<float>>& vectors,
    std::size_t k,
    const InitialClusteringConfig& cfg,
    std::uint32_t seed,
    KMeansModel* out_model);

Status select_k_binary_elbow(
    const std::vector<std::vector<float>>& vectors,
    const IdEstimateRange& id_range,
    const InitialClusteringConfig& cfg,
    KMeansModel* out_best_model,
    ElbowSelection* out_selection);

Status evaluate_stability(
    const std::vector<std::vector<float>>& vectors,
    std::size_t chosen_k,
    const InitialClusteringConfig& cfg,
    StabilityMetrics* out_metrics);

bool cuda_dot_products_available();
bool cuda_assignment_kernels_available();
Status cuda_compute_dot_products(
    const std::vector<float>& vectors_row_major,
    const std::vector<float>& centroids_row_major,
    std::size_t n_vectors,
    std::size_t k_centroids,
    std::size_t dim,
    std::vector<float>* out_scores_row_major,
    bool* out_tensor_core_enabled,
    std::string* out_backend_name);
Status cuda_assign_top1_labels(
    const std::vector<float>& scores_row_major,
    std::size_t n_vectors,
    std::size_t k_centroids,
    std::vector<std::uint32_t>* out_labels,
    std::vector<float>* out_best_scores);
Status cuda_reduce_centroids_top1(
    const std::vector<float>& vectors_row_major,
    const std::vector<std::uint32_t>& labels,
    std::size_t n_vectors,
    std::size_t k_centroids,
    std::size_t dim,
    std::vector<float>* out_centroids_row_major);

}  // namespace vector_db

