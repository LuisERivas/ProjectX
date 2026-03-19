#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vector_db_v3::codec {

struct CommonHeader {
    std::uint16_t schema_version = 0;
    std::uint16_t record_type = 0;
    std::uint32_t record_count = 0;
    std::uint32_t payload_bytes = 0;
    std::uint32_t checksum_crc32 = 0;
};

struct IdEstimateRow {
    std::uint32_t k_min = 0;
    std::uint32_t k_max = 0;
    std::uint16_t id_estimate_method = 0;
    std::uint16_t reserved = 0;
};

enum class ProbePhase : std::uint8_t {
    Coarse = 1,
    Fine = 2,
};

struct ElbowTraceRow {
    std::uint32_t k_value = 0;
    float objective_value = 0.0f;
    ProbePhase probe_phase = ProbePhase::Coarse;
    std::array<std::uint8_t, 3> reserved{0, 0, 0};
};

enum class StabilityStatusCode : std::uint16_t {
    Unknown = 0,
    Ok = 1,
    Warning = 2,
    Fail = 3,
};

struct StabilityReportRow {
    StabilityStatusCode status_code = StabilityStatusCode::Unknown;
    std::uint16_t reserved = 0;
    float mean_nmi = 0.0f;
    float std_nmi = 0.0f;
    float mean_jaccard = 0.0f;
    float mean_centroid_drift = 0.0f;
};

struct TopAssignmentRow {
    std::uint64_t embedding_id = 0;
    std::uint32_t top_centroid_numeric_id = 0;
};

struct TopCentroidRow {
    std::uint32_t top_centroid_numeric_id = 0;
    std::array<float, 1024> centroid_vector{};
};

struct MidAssignmentRow {
    std::uint64_t embedding_id = 0;
    std::uint32_t mid_centroid_numeric_id = 0;
    std::uint32_t parent_top_centroid_numeric_id = 0;
};

struct FinalAssignmentRow {
    std::uint64_t embedding_id = 0;
    std::uint32_t final_cluster_numeric_id = 0;
};

enum class StageLevel : std::uint8_t {
    Top = 1,
    Mid = 2,
    Lower = 3,
};

enum class GateDecision : std::uint8_t {
    NotApplicable = 0,
    Continue = 1,
    Stop = 2,
};

struct KSearchBoundsBatchRow {
    StageLevel stage_level = StageLevel::Top;
    GateDecision gate_decision = GateDecision::NotApplicable;
    std::uint16_t reserved = 0;
    std::uint32_t source_numeric_id = 0;
    std::uint32_t k_min = 0;
    std::uint32_t k_max = 0;
    std::uint32_t chosen_k = 0;
    std::uint32_t dataset_size = 0;
};

struct PostClusterMembershipRow {
    std::uint64_t embedding_id = 0;
    std::uint32_t top_centroid_numeric_id = 0;
    std::uint32_t mid_centroid_numeric_id = 0;
    std::uint32_t lower_centroid_numeric_id = 0;
    std::uint32_t final_cluster_numeric_id = 0;
};

struct PrecisionAlignmentResult {
    bool pass = false;
    std::size_t mismatch_count = 0;
    std::string reason;
};

enum class EmbeddingShardValueType : std::uint16_t {
    FP32 = 1,
    FP16 = 2,
    INT8Sym = 3,
};

struct EmbeddingShardHeader {
    std::uint32_t magic = 0;
    std::uint16_t schema_version = 1;
    EmbeddingShardValueType value_type = EmbeddingShardValueType::FP32;
    std::uint32_t record_size_bytes = 0;
    std::uint64_t record_count = 0;
    std::uint32_t reserved = 0;
};

struct PrecisionShardSelectionInfo {
    bool observed = false;
    std::string source_embedding_artifact = "embeddings_fp32.bin";
    std::string compute_precision = "fp32";
    std::string alignment_check_status = "pass";
    std::size_t alignment_mismatch_count = 0;
    std::string alignment_failure_reason;
    std::string fallback_reason;
};

constexpr std::size_t kCommonHeaderSize = 16;
constexpr std::size_t kIdEstimateRecordSize = 12;
constexpr std::size_t kElbowTraceRecordSize = 12;
constexpr std::size_t kStabilityReportRecordSize = 20;
constexpr std::size_t kTopAssignmentRecordSize = 12;
constexpr std::size_t kTopCentroidRecordSize = 4 + (1024 * sizeof(float));
constexpr std::size_t kMidAssignmentRecordSize = 16;
constexpr std::size_t kFinalAssignmentRecordSize = 12;
constexpr std::size_t kKSearchBoundsBatchRecordSize = 24;
constexpr std::size_t kPostClusterMembershipRecordSize = 24;
constexpr std::size_t kEmbeddingShardHeaderSize = 24;
constexpr std::size_t kEmbeddingShardRecordSizeFp32 = 8 + (1024 * sizeof(float));
constexpr std::size_t kEmbeddingShardRecordSizeFp16 = 8 + (1024 * sizeof(std::uint16_t));
constexpr std::size_t kEmbeddingShardRecordSizeInt8Sym = 8 + 4 + 1024;

}  // namespace vector_db_v3::codec
