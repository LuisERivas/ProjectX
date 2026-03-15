#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "vector_db_v3/codec/types.hpp"
#include "vector_db_v3/status.hpp"

namespace vector_db_v3::codec {

Status encode_common_header(const CommonHeader& hdr, std::vector<std::uint8_t>* out);
Status decode_common_header(const std::vector<std::uint8_t>& bytes, CommonHeader* out);
Status encode_header_plus_payload(
    CommonHeader hdr,
    const std::vector<std::uint8_t>& payload,
    std::vector<std::uint8_t>* out);
Status decode_header_plus_payload(
    const std::vector<std::uint8_t>& bytes,
    CommonHeader* hdr,
    std::vector<std::uint8_t>* payload);

Status encode_id_estimate(const IdEstimateRow& row, std::vector<std::uint8_t>* out);
Status decode_id_estimate(const std::vector<std::uint8_t>& bytes, IdEstimateRow* out);

Status encode_elbow_trace(const std::vector<ElbowTraceRow>& rows, std::vector<std::uint8_t>* out);
Status decode_elbow_trace(const std::vector<std::uint8_t>& bytes, std::vector<ElbowTraceRow>* out);

Status encode_stability_report(const StabilityReportRow& row, std::vector<std::uint8_t>* out);
Status decode_stability_report(const std::vector<std::uint8_t>& bytes, StabilityReportRow* out);

Status encode_top_assignments(const std::vector<TopAssignmentRow>& rows, std::vector<std::uint8_t>* out);
Status decode_top_assignments(const std::vector<std::uint8_t>& bytes, std::vector<TopAssignmentRow>* out);
Status encode_top_centroids(const std::vector<TopCentroidRow>& rows, std::vector<std::uint8_t>* out);
Status decode_top_centroids(const std::vector<std::uint8_t>& bytes, std::vector<TopCentroidRow>* out);
Status encode_mid_assignments(const std::vector<MidAssignmentRow>& rows, std::vector<std::uint8_t>* out);
Status decode_mid_assignments(const std::vector<std::uint8_t>& bytes, std::vector<MidAssignmentRow>* out);
Status encode_final_assignments(const std::vector<FinalAssignmentRow>& rows, std::vector<std::uint8_t>* out);
Status decode_final_assignments(const std::vector<std::uint8_t>& bytes, std::vector<FinalAssignmentRow>* out);

Status encode_k_search_bounds_batch(
    const std::vector<KSearchBoundsBatchRow>& rows,
    std::vector<std::uint8_t>* out);
Status decode_k_search_bounds_batch(
    const std::vector<std::uint8_t>& bytes,
    std::vector<KSearchBoundsBatchRow>* out);

Status encode_post_cluster_membership(
    const std::vector<PostClusterMembershipRow>& rows,
    std::vector<std::uint8_t>* out);
Status decode_post_cluster_membership(
    const std::vector<std::uint8_t>& bytes,
    std::vector<PostClusterMembershipRow>* out);

Status write_id_estimate_file(const std::filesystem::path& path, const IdEstimateRow& row);
Status read_id_estimate_file(const std::filesystem::path& path, IdEstimateRow* row);
Status write_elbow_trace_file(const std::filesystem::path& path, const std::vector<ElbowTraceRow>& rows);
Status read_elbow_trace_file(const std::filesystem::path& path, std::vector<ElbowTraceRow>* rows);
Status write_stability_report_file(const std::filesystem::path& path, const StabilityReportRow& row);
Status read_stability_report_file(const std::filesystem::path& path, StabilityReportRow* row);
Status write_top_assignments_file(const std::filesystem::path& path, const std::vector<TopAssignmentRow>& rows);
Status read_top_assignments_file(const std::filesystem::path& path, std::vector<TopAssignmentRow>* rows);
Status write_top_centroids_file(const std::filesystem::path& path, const std::vector<TopCentroidRow>& rows);
Status read_top_centroids_file(const std::filesystem::path& path, std::vector<TopCentroidRow>* rows);
Status write_cluster_manifest_file(const std::filesystem::path& path, const std::vector<std::uint8_t>& payload);
Status read_cluster_manifest_file(
    const std::filesystem::path& path,
    CommonHeader* header,
    std::vector<std::uint8_t>* payload);
Status write_mid_assignments_file(const std::filesystem::path& path, const std::vector<MidAssignmentRow>& rows);
Status read_mid_assignments_file(const std::filesystem::path& path, std::vector<MidAssignmentRow>* rows);
Status write_final_assignments_file(const std::filesystem::path& path, const std::vector<FinalAssignmentRow>& rows);
Status read_final_assignments_file(const std::filesystem::path& path, std::vector<FinalAssignmentRow>* rows);
Status write_k_search_bounds_batch_file(
    const std::filesystem::path& path,
    const std::vector<KSearchBoundsBatchRow>& rows);
Status read_k_search_bounds_batch_file(
    const std::filesystem::path& path,
    std::vector<KSearchBoundsBatchRow>* rows);
Status write_post_cluster_membership_file(
    const std::filesystem::path& path,
    const std::vector<PostClusterMembershipRow>& rows);
Status read_post_cluster_membership_file(
    const std::filesystem::path& path,
    std::vector<PostClusterMembershipRow>* rows);

Status validate_id_estimate(const IdEstimateRow& row);
Status validate_elbow_trace(const std::vector<ElbowTraceRow>& rows, std::optional<std::uint32_t> chosen_k = std::nullopt);
Status validate_stability_report(const StabilityReportRow& row);
Status validate_top_assignments(const std::vector<TopAssignmentRow>& rows);
Status validate_top_centroids(const std::vector<TopCentroidRow>& rows);
Status validate_mid_assignments(const std::vector<MidAssignmentRow>& rows);
Status validate_final_assignments(const std::vector<FinalAssignmentRow>& rows);
Status validate_k_search_bounds_batch(const std::vector<KSearchBoundsBatchRow>& rows);
Status validate_post_cluster_membership(
    const std::vector<PostClusterMembershipRow>& rows,
    bool allow_lower_sentinel = true);

PrecisionAlignmentResult evaluate_precision_id_alignment(
    const std::vector<std::uint64_t>& fp32_ids,
    const std::optional<std::vector<std::uint64_t>>& fp16_ids,
    const std::vector<std::vector<std::uint64_t>>& int8_variants);
Status validate_precision_id_alignment(
    const std::vector<std::uint64_t>& fp32_ids,
    const std::optional<std::vector<std::uint64_t>>& fp16_ids,
    const std::vector<std::vector<std::uint64_t>>& int8_variants);

}  // namespace vector_db_v3::codec
