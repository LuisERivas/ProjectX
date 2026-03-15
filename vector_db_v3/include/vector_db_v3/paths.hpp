#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace vector_db_v3::paths {

inline std::filesystem::path manifest(const std::filesystem::path& data_dir) {
    return data_dir / "manifest.json";
}

inline std::filesystem::path wal(const std::filesystem::path& data_dir) {
    return data_dir / "wal.log";
}

inline std::filesystem::path segments_dir(const std::filesystem::path& data_dir) {
    return data_dir / "segments";
}

inline std::filesystem::path checkpoint_bin(
    const std::filesystem::path& data_dir,
    std::uint64_t checkpoint_lsn) {
    return segments_dir(data_dir) / ("checkpoint_" + std::to_string(checkpoint_lsn) + ".bin");
}

inline std::filesystem::path checkpoint_temp_bin(const std::filesystem::path& data_dir) {
    return segments_dir(data_dir) / "checkpoint_tmp.bin";
}

inline std::filesystem::path clusters_current_dir(const std::filesystem::path& data_dir) {
    return data_dir / "clusters" / "current";
}

inline std::filesystem::path id_estimate_bin(const std::filesystem::path& data_dir) {
    return clusters_current_dir(data_dir) / "id_estimate.bin";
}

inline std::filesystem::path elbow_trace_bin(const std::filesystem::path& data_dir) {
    return clusters_current_dir(data_dir) / "elbow_trace.bin";
}

inline std::filesystem::path centroids_bin(const std::filesystem::path& data_dir) {
    return clusters_current_dir(data_dir) / "centroids.bin";
}

inline std::filesystem::path top_assignments_bin(const std::filesystem::path& data_dir) {
    return clusters_current_dir(data_dir) / "assignments.bin";
}

inline std::filesystem::path stability_report_bin(const std::filesystem::path& data_dir) {
    return clusters_current_dir(data_dir) / "stability_report.bin";
}

inline std::filesystem::path cluster_manifest_bin(const std::filesystem::path& data_dir) {
    return clusters_current_dir(data_dir) / "cluster_manifest.bin";
}

inline std::filesystem::path mid_layer_dir(const std::filesystem::path& data_dir) {
    return clusters_current_dir(data_dir) / "mid_layer_clustering";
}

inline std::filesystem::path mid_layer_clustering_bin(const std::filesystem::path& data_dir) {
    return mid_layer_dir(data_dir) / "MID_LAYER_CLUSTERING.bin";
}

inline std::filesystem::path mid_layer_assignments_bin(const std::filesystem::path& data_dir) {
    return mid_layer_dir(data_dir) / "assignments.bin";
}

inline std::filesystem::path lower_layer_dir(const std::filesystem::path& data_dir) {
    return clusters_current_dir(data_dir) / "lower_layer_clustering";
}

inline std::filesystem::path lower_layer_clustering_bin(const std::filesystem::path& data_dir) {
    return lower_layer_dir(data_dir) / "LOWER_LAYER_CLUSTERING.bin";
}

inline std::filesystem::path final_layer_dir(const std::filesystem::path& data_dir) {
    return clusters_current_dir(data_dir) / "final_layer_clustering";
}

inline std::filesystem::path final_layer_clusters_bin(const std::filesystem::path& data_dir) {
    return final_layer_dir(data_dir) / "FINAL_LAYER_CLUSTERS.bin";
}

inline std::filesystem::path final_cluster_dir(
    const std::filesystem::path& data_dir,
    std::uint32_t cluster_id) {
    return final_layer_dir(data_dir) / ("final_cluster_" + std::to_string(cluster_id));
}

inline std::filesystem::path final_cluster_manifest_bin(
    const std::filesystem::path& data_dir,
    std::uint32_t cluster_id) {
    return final_cluster_dir(data_dir, cluster_id) / "manifest.bin";
}

inline std::filesystem::path final_cluster_assignments_bin(
    const std::filesystem::path& data_dir,
    std::uint32_t cluster_id) {
    return final_cluster_dir(data_dir, cluster_id) / "assignments.bin";
}

inline std::filesystem::path final_cluster_summary_bin(
    const std::filesystem::path& data_dir,
    std::uint32_t cluster_id) {
    return final_cluster_dir(data_dir, cluster_id) / "cluster_summary.bin";
}

inline std::filesystem::path k_search_bounds_batch_bin(const std::filesystem::path& data_dir) {
    return clusters_current_dir(data_dir) / "k_search_bounds_batch.bin";
}

inline std::filesystem::path post_cluster_membership_bin(const std::filesystem::path& data_dir) {
    return clusters_current_dir(data_dir) / "post_cluster_membership.bin";
}

inline std::filesystem::path telemetry_baseline_jsonl(const std::filesystem::path& data_dir) {
    return clusters_current_dir(data_dir) / "telemetry_stage_baseline.jsonl";
}

}  // namespace vector_db_v3::paths
