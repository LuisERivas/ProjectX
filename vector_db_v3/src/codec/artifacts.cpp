#include "vector_db_v3/codec/artifacts.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "vector_db_v3/codec/checksum.hpp"
#include "vector_db_v3/codec/endian.hpp"
#include "vector_db_v3/codec/io.hpp"

namespace vector_db_v3::codec {

namespace {

template <typename T>
Status ensure_out(T* ptr, const char* where) {
    if (ptr == nullptr) {
        return Status::Error(std::string(where) + ": output pointer is null");
    }
    return Status::Ok();
}

template <typename Row, typename WriteFn>
Status encode_rows(
    const std::vector<Row>& rows,
    std::size_t record_size,
    WriteFn&& write_fn,
    std::vector<std::uint8_t>* out) {
    const Status guard = ensure_out(out, "encode_rows");
    if (!guard.ok) {
        return guard;
    }
    out->assign(rows.size() * record_size, 0U);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        write_fn(rows[i], out->data() + i * record_size);
    }
    return Status::Ok();
}

template <typename Row, typename ReadFn>
Status decode_rows(
    const std::vector<std::uint8_t>& bytes,
    std::size_t record_size,
    const std::string& artifact_name,
    ReadFn&& read_fn,
    std::vector<Row>* out) {
    const Status guard = ensure_out(out, "decode_rows");
    if (!guard.ok) {
        return guard;
    }
    const Status sized = validate_exact_size_multiple(bytes.size(), record_size, artifact_name);
    if (!sized.ok) {
        return sized;
    }
    const std::size_t count = bytes.size() / record_size;
    out->assign(count, Row{});
    for (std::size_t i = 0; i < count; ++i) {
        read_fn(bytes.data() + i * record_size, &(*out)[i]);
    }
    return Status::Ok();
}

template <typename Row>
Status validate_sorted_unique_embedding_ids(
    const std::vector<Row>& rows,
    const std::string& artifact_name) {
    for (std::size_t i = 1; i < rows.size(); ++i) {
        if (rows[i - 1].embedding_id > rows[i].embedding_id) {
            return Status::Error(artifact_name + ": embedding_id not sorted ascending");
        }
        if (rows[i - 1].embedding_id == rows[i].embedding_id) {
            return Status::Error(artifact_name + ": duplicate embedding_id");
        }
    }
    return Status::Ok();
}

Status write_bytes_with_info(
    const std::filesystem::path& path,
    const std::vector<std::uint8_t>& bytes,
    WriteArtifactInfo* out_info) {
    const Status write_st = write_atomic_bytes(path, bytes);
    if (!write_st.ok) {
        return write_st;
    }
    if (out_info != nullptr) {
        out_info->bytes_written = bytes.size();
        out_info->checksum_sha256 = sha256_hex(bytes);
    }
    return Status::Ok();
}

template <typename T, typename DecFn>
Status read_single_fixed_record(
    const std::filesystem::path& path,
    T* row,
    DecFn&& dec_fn) {
    std::vector<std::uint8_t> bytes;
    const Status rd = read_file_bytes(path, &bytes);
    if (!rd.ok) {
        return rd;
    }
    return dec_fn(bytes, row);
}

template <typename T, typename EncFn>
Status write_rows_record_with_info(
    const std::filesystem::path& path,
    const std::vector<T>& rows,
    EncFn&& enc_fn,
    WriteArtifactInfo* out_info) {
    std::vector<std::uint8_t> bytes;
    const Status enc = enc_fn(rows, &bytes);
    if (!enc.ok) {
        return enc;
    }
    return write_bytes_with_info(path, bytes, out_info);
}

template <typename T, typename DecFn>
Status read_rows_record(
    const std::filesystem::path& path,
    std::vector<T>* rows,
    DecFn&& dec_fn) {
    std::vector<std::uint8_t> bytes;
    const Status rd = read_file_bytes(path, &bytes);
    if (!rd.ok) {
        return rd;
    }
    return dec_fn(bytes, rows);
}

constexpr std::uint32_t kEmbeddingShardMagicFp32 = 0x33465045U;     // EFP3
constexpr std::uint32_t kEmbeddingShardMagicFp16 = 0x36465045U;     // EFP6
constexpr std::uint32_t kEmbeddingShardMagicInt8Sym = 0x38535045U;  // EPS8

std::uint16_t fp32_to_fp16_bits(float value) {
    std::uint32_t x = 0;
    std::memcpy(&x, &value, sizeof(float));
    const std::uint32_t sign = (x >> 16U) & 0x8000U;
    const std::int32_t exp = static_cast<std::int32_t>((x >> 23U) & 0xFFU) - 127 + 15;
    std::uint32_t mant = x & 0x007FFFFFU;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mant = (mant | 0x00800000U) >> static_cast<std::uint32_t>(1 - exp);
        return static_cast<std::uint16_t>(sign | ((mant + 0x00001000U) >> 13U));
    }
    if (exp >= 0x1F) {
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10U) | ((mant + 0x00001000U) >> 13U));
}

float fp16_bits_to_fp32(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
    const std::uint32_t exp = (h >> 10U) & 0x1FU;
    const std::uint32_t mant = h & 0x03FFU;

    std::uint32_t out = 0;
    if (exp == 0U) {
        if (mant == 0U) {
            out = sign;
        } else {
            std::uint32_t m = mant;
            std::int32_t e = -1;
            while ((m & 0x0400U) == 0U) {
                m <<= 1U;
                --e;
            }
            m &= 0x03FFU;
            const std::uint32_t exp32 = static_cast<std::uint32_t>(127 - 15 + 1 + e);
            out = sign | (exp32 << 23U) | (m << 13U);
        }
    } else if (exp == 0x1FU) {
        out = sign | 0x7F800000U | (mant << 13U);
    } else {
        const std::uint32_t exp32 = exp + (127U - 15U);
        out = sign | (exp32 << 23U) | (mant << 13U);
    }
    float f = 0.0f;
    std::memcpy(&f, &out, sizeof(float));
    return f;
}

}  // namespace

Status encode_common_header(const CommonHeader& hdr, std::vector<std::uint8_t>* out) {
    const Status guard = ensure_out(out, "encode_common_header");
    if (!guard.ok) {
        return guard;
    }
    out->assign(kCommonHeaderSize, 0U);
    store_le_u16(out->data() + 0U, hdr.schema_version);
    store_le_u16(out->data() + 2U, hdr.record_type);
    store_le_u32(out->data() + 4U, hdr.record_count);
    store_le_u32(out->data() + 8U, hdr.payload_bytes);
    store_le_u32(out->data() + 12U, hdr.checksum_crc32);
    return Status::Ok();
}

Status decode_common_header(const std::vector<std::uint8_t>& bytes, CommonHeader* out) {
    const Status guard = ensure_out(out, "decode_common_header");
    if (!guard.ok) {
        return guard;
    }
    if (bytes.size() != kCommonHeaderSize) {
        return Status::Error("decode_common_header: expected 16 bytes");
    }
    out->schema_version = load_le_u16(bytes.data() + 0U);
    out->record_type = load_le_u16(bytes.data() + 2U);
    out->record_count = load_le_u32(bytes.data() + 4U);
    out->payload_bytes = load_le_u32(bytes.data() + 8U);
    out->checksum_crc32 = load_le_u32(bytes.data() + 12U);
    return Status::Ok();
}

Status encode_header_plus_payload(
    CommonHeader hdr,
    const std::vector<std::uint8_t>& payload,
    std::vector<std::uint8_t>* out) {
    const Status guard = ensure_out(out, "encode_header_plus_payload");
    if (!guard.ok) {
        return guard;
    }
    hdr.payload_bytes = static_cast<std::uint32_t>(payload.size());
    hdr.checksum_crc32 = crc32(payload);
    std::vector<std::uint8_t> header_bytes;
    const Status h = encode_common_header(hdr, &header_bytes);
    if (!h.ok) {
        return h;
    }
    out->assign(header_bytes.begin(), header_bytes.end());
    out->insert(out->end(), payload.begin(), payload.end());
    return Status::Ok();
}

Status decode_header_plus_payload(
    const std::vector<std::uint8_t>& bytes,
    CommonHeader* hdr,
    std::vector<std::uint8_t>* payload) {
    const Status guard1 = ensure_out(hdr, "decode_header_plus_payload");
    if (!guard1.ok) {
        return guard1;
    }
    const Status guard2 = ensure_out(payload, "decode_header_plus_payload");
    if (!guard2.ok) {
        return guard2;
    }
    if (bytes.size() < kCommonHeaderSize) {
        return Status::Error("decode_header_plus_payload: bytes shorter than header");
    }
    std::vector<std::uint8_t> header(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(kCommonHeaderSize));
    const Status hs = decode_common_header(header, hdr);
    if (!hs.ok) {
        return hs;
    }
    const std::size_t payload_size = bytes.size() - kCommonHeaderSize;
    if (payload_size != static_cast<std::size_t>(hdr->payload_bytes)) {
        return Status::Error("decode_header_plus_payload: payload_bytes mismatch");
    }
    payload->assign(bytes.begin() + static_cast<std::ptrdiff_t>(kCommonHeaderSize), bytes.end());
    if (crc32(*payload) != hdr->checksum_crc32) {
        return Status::Error("decode_header_plus_payload: checksum mismatch");
    }
    return Status::Ok();
}

Status encode_id_estimate(const IdEstimateRow& row, std::vector<std::uint8_t>* out) {
    const Status guard = ensure_out(out, "encode_id_estimate");
    if (!guard.ok) {
        return guard;
    }
    out->assign(kIdEstimateRecordSize, 0U);
    store_le_u32(out->data() + 0U, row.k_min);
    store_le_u32(out->data() + 4U, row.k_max);
    store_le_u16(out->data() + 8U, row.id_estimate_method);
    store_le_u16(out->data() + 10U, row.reserved);
    return Status::Ok();
}

Status decode_id_estimate(const std::vector<std::uint8_t>& bytes, IdEstimateRow* out) {
    const Status guard = ensure_out(out, "decode_id_estimate");
    if (!guard.ok) {
        return guard;
    }
    if (bytes.size() != kIdEstimateRecordSize) {
        return Status::Error("decode_id_estimate: expected 12 bytes");
    }
    out->k_min = load_le_u32(bytes.data() + 0U);
    out->k_max = load_le_u32(bytes.data() + 4U);
    out->id_estimate_method = load_le_u16(bytes.data() + 8U);
    out->reserved = load_le_u16(bytes.data() + 10U);
    return validate_id_estimate(*out);
}

Status encode_elbow_trace(const std::vector<ElbowTraceRow>& rows, std::vector<std::uint8_t>* out) {
    return encode_rows(
        rows,
        kElbowTraceRecordSize,
        [](const ElbowTraceRow& row, std::uint8_t* dst) {
            store_le_u32(dst + 0U, row.k_value);
            store_le_f32(dst + 4U, row.objective_value);
            dst[8U] = static_cast<std::uint8_t>(row.probe_phase);
            dst[9U] = row.reserved[0];
            dst[10U] = row.reserved[1];
            dst[11U] = row.reserved[2];
        },
        out);
}

Status decode_elbow_trace(const std::vector<std::uint8_t>& bytes, std::vector<ElbowTraceRow>* out) {
    const Status s = decode_rows(
        bytes,
        kElbowTraceRecordSize,
        "elbow_trace.bin",
        [](const std::uint8_t* src, ElbowTraceRow* row) {
            row->k_value = load_le_u32(src + 0U);
            row->objective_value = load_le_f32(src + 4U);
            row->probe_phase = static_cast<ProbePhase>(src[8U]);
            row->reserved = {src[9U], src[10U], src[11U]};
        },
        out);
    if (!s.ok) {
        return s;
    }
    return validate_elbow_trace(*out);
}

Status encode_stability_report(const StabilityReportRow& row, std::vector<std::uint8_t>* out) {
    const Status guard = ensure_out(out, "encode_stability_report");
    if (!guard.ok) {
        return guard;
    }
    out->assign(kStabilityReportRecordSize, 0U);
    store_le_u16(out->data() + 0U, static_cast<std::uint16_t>(row.status_code));
    store_le_u16(out->data() + 2U, row.reserved);
    store_le_f32(out->data() + 4U, row.mean_nmi);
    store_le_f32(out->data() + 8U, row.std_nmi);
    store_le_f32(out->data() + 12U, row.mean_jaccard);
    store_le_f32(out->data() + 16U, row.mean_centroid_drift);
    return Status::Ok();
}

Status decode_stability_report(const std::vector<std::uint8_t>& bytes, StabilityReportRow* out) {
    const Status guard = ensure_out(out, "decode_stability_report");
    if (!guard.ok) {
        return guard;
    }
    if (bytes.size() != kStabilityReportRecordSize) {
        return Status::Error("decode_stability_report: expected 20 bytes");
    }
    out->status_code = static_cast<StabilityStatusCode>(load_le_u16(bytes.data() + 0U));
    out->reserved = load_le_u16(bytes.data() + 2U);
    out->mean_nmi = load_le_f32(bytes.data() + 4U);
    out->std_nmi = load_le_f32(bytes.data() + 8U);
    out->mean_jaccard = load_le_f32(bytes.data() + 12U);
    out->mean_centroid_drift = load_le_f32(bytes.data() + 16U);
    return validate_stability_report(*out);
}

Status encode_top_assignments(const std::vector<TopAssignmentRow>& rows, std::vector<std::uint8_t>* out) {
    return encode_rows(
        rows,
        kTopAssignmentRecordSize,
        [](const TopAssignmentRow& row, std::uint8_t* dst) {
            store_le_u64(dst + 0U, row.embedding_id);
            store_le_u32(dst + 8U, row.top_centroid_numeric_id);
        },
        out);
}

Status decode_top_assignments(const std::vector<std::uint8_t>& bytes, std::vector<TopAssignmentRow>* out) {
    const Status s = decode_rows(
        bytes,
        kTopAssignmentRecordSize,
        "assignments.bin(top)",
        [](const std::uint8_t* src, TopAssignmentRow* row) {
            row->embedding_id = load_le_u64(src + 0U);
            row->top_centroid_numeric_id = load_le_u32(src + 8U);
        },
        out);
    if (!s.ok) {
        return s;
    }
    return validate_top_assignments(*out);
}

Status encode_top_centroids(const std::vector<TopCentroidRow>& rows, std::vector<std::uint8_t>* out) {
    return encode_rows(
        rows,
        kTopCentroidRecordSize,
        [](const TopCentroidRow& row, std::uint8_t* dst) {
            store_le_u32(dst + 0U, row.top_centroid_numeric_id);
            for (std::size_t i = 0; i < row.centroid_vector.size(); ++i) {
                store_le_f32(dst + 4U + (i * sizeof(float)), row.centroid_vector[i]);
            }
        },
        out);
}

Status decode_top_centroids(const std::vector<std::uint8_t>& bytes, std::vector<TopCentroidRow>* out) {
    const Status s = decode_rows(
        bytes,
        kTopCentroidRecordSize,
        "centroids.bin(top)",
        [](const std::uint8_t* src, TopCentroidRow* row) {
            row->top_centroid_numeric_id = load_le_u32(src + 0U);
            for (std::size_t i = 0; i < row->centroid_vector.size(); ++i) {
                row->centroid_vector[i] = load_le_f32(src + 4U + (i * sizeof(float)));
            }
        },
        out);
    if (!s.ok) {
        return s;
    }
    return validate_top_centroids(*out);
}

Status encode_mid_assignments(const std::vector<MidAssignmentRow>& rows, std::vector<std::uint8_t>* out) {
    return encode_rows(
        rows,
        kMidAssignmentRecordSize,
        [](const MidAssignmentRow& row, std::uint8_t* dst) {
            store_le_u64(dst + 0U, row.embedding_id);
            store_le_u32(dst + 8U, row.mid_centroid_numeric_id);
            store_le_u32(dst + 12U, row.parent_top_centroid_numeric_id);
        },
        out);
}

Status decode_mid_assignments(const std::vector<std::uint8_t>& bytes, std::vector<MidAssignmentRow>* out) {
    const Status s = decode_rows(
        bytes,
        kMidAssignmentRecordSize,
        "assignments.bin(mid)",
        [](const std::uint8_t* src, MidAssignmentRow* row) {
            row->embedding_id = load_le_u64(src + 0U);
            row->mid_centroid_numeric_id = load_le_u32(src + 8U);
            row->parent_top_centroid_numeric_id = load_le_u32(src + 12U);
        },
        out);
    if (!s.ok) {
        return s;
    }
    return validate_mid_assignments(*out);
}

Status encode_final_assignments(const std::vector<FinalAssignmentRow>& rows, std::vector<std::uint8_t>* out) {
    return encode_rows(
        rows,
        kFinalAssignmentRecordSize,
        [](const FinalAssignmentRow& row, std::uint8_t* dst) {
            store_le_u64(dst + 0U, row.embedding_id);
            store_le_u32(dst + 8U, row.final_cluster_numeric_id);
        },
        out);
}

Status decode_final_assignments(const std::vector<std::uint8_t>& bytes, std::vector<FinalAssignmentRow>* out) {
    const Status s = decode_rows(
        bytes,
        kFinalAssignmentRecordSize,
        "assignments.bin(final)",
        [](const std::uint8_t* src, FinalAssignmentRow* row) {
            row->embedding_id = load_le_u64(src + 0U);
            row->final_cluster_numeric_id = load_le_u32(src + 8U);
        },
        out);
    if (!s.ok) {
        return s;
    }
    return validate_final_assignments(*out);
}

Status encode_k_search_bounds_batch(
    const std::vector<KSearchBoundsBatchRow>& rows,
    std::vector<std::uint8_t>* out) {
    return encode_rows(
        rows,
        kKSearchBoundsBatchRecordSize,
        [](const KSearchBoundsBatchRow& row, std::uint8_t* dst) {
            dst[0U] = static_cast<std::uint8_t>(row.stage_level);
            dst[1U] = static_cast<std::uint8_t>(row.gate_decision);
            store_le_u16(dst + 2U, row.reserved);
            store_le_u32(dst + 4U, row.source_numeric_id);
            store_le_u32(dst + 8U, row.k_min);
            store_le_u32(dst + 12U, row.k_max);
            store_le_u32(dst + 16U, row.chosen_k);
            store_le_u32(dst + 20U, row.dataset_size);
        },
        out);
}

Status decode_k_search_bounds_batch(
    const std::vector<std::uint8_t>& bytes,
    std::vector<KSearchBoundsBatchRow>* out) {
    const Status s = decode_rows(
        bytes,
        kKSearchBoundsBatchRecordSize,
        "k_search_bounds_batch.bin",
        [](const std::uint8_t* src, KSearchBoundsBatchRow* row) {
            row->stage_level = static_cast<StageLevel>(src[0U]);
            row->gate_decision = static_cast<GateDecision>(src[1U]);
            row->reserved = load_le_u16(src + 2U);
            row->source_numeric_id = load_le_u32(src + 4U);
            row->k_min = load_le_u32(src + 8U);
            row->k_max = load_le_u32(src + 12U);
            row->chosen_k = load_le_u32(src + 16U);
            row->dataset_size = load_le_u32(src + 20U);
        },
        out);
    if (!s.ok) {
        return s;
    }
    return validate_k_search_bounds_batch(*out);
}

Status encode_post_cluster_membership(
    const std::vector<PostClusterMembershipRow>& rows,
    std::vector<std::uint8_t>* out) {
    return encode_rows(
        rows,
        kPostClusterMembershipRecordSize,
        [](const PostClusterMembershipRow& row, std::uint8_t* dst) {
            store_le_u64(dst + 0U, row.embedding_id);
            store_le_u32(dst + 8U, row.top_centroid_numeric_id);
            store_le_u32(dst + 12U, row.mid_centroid_numeric_id);
            store_le_u32(dst + 16U, row.lower_centroid_numeric_id);
            store_le_u32(dst + 20U, row.final_cluster_numeric_id);
        },
        out);
}

Status decode_post_cluster_membership(
    const std::vector<std::uint8_t>& bytes,
    std::vector<PostClusterMembershipRow>* out) {
    const Status s = decode_rows(
        bytes,
        kPostClusterMembershipRecordSize,
        "post_cluster_membership.bin",
        [](const std::uint8_t* src, PostClusterMembershipRow* row) {
            row->embedding_id = load_le_u64(src + 0U);
            row->top_centroid_numeric_id = load_le_u32(src + 8U);
            row->mid_centroid_numeric_id = load_le_u32(src + 12U);
            row->lower_centroid_numeric_id = load_le_u32(src + 16U);
            row->final_cluster_numeric_id = load_le_u32(src + 20U);
        },
        out);
    if (!s.ok) {
        return s;
    }
    return validate_post_cluster_membership(*out, true);
}

Status write_id_estimate_file(const std::filesystem::path& path, const IdEstimateRow& row) {
    return write_id_estimate_file_with_info(path, row, nullptr);
}

Status write_id_estimate_file_with_info(
    const std::filesystem::path& path,
    const IdEstimateRow& row,
    WriteArtifactInfo* out_info) {
    std::vector<std::uint8_t> bytes;
    const Status enc = encode_id_estimate(row, &bytes);
    if (!enc.ok) {
        return enc;
    }
    return write_bytes_with_info(path, bytes, out_info);
}

Status read_id_estimate_file(const std::filesystem::path& path, IdEstimateRow* row) {
    return read_single_fixed_record(path, row, decode_id_estimate);
}

Status write_elbow_trace_file(const std::filesystem::path& path, const std::vector<ElbowTraceRow>& rows) {
    return write_elbow_trace_file_with_info(path, rows, nullptr);
}

Status write_elbow_trace_file_with_info(
    const std::filesystem::path& path,
    const std::vector<ElbowTraceRow>& rows,
    WriteArtifactInfo* out_info) {
    return write_rows_record_with_info(path, rows, encode_elbow_trace, out_info);
}

Status read_elbow_trace_file(const std::filesystem::path& path, std::vector<ElbowTraceRow>* rows) {
    return read_rows_record(path, rows, decode_elbow_trace);
}

Status write_stability_report_file(const std::filesystem::path& path, const StabilityReportRow& row) {
    return write_stability_report_file_with_info(path, row, nullptr);
}

Status write_stability_report_file_with_info(
    const std::filesystem::path& path,
    const StabilityReportRow& row,
    WriteArtifactInfo* out_info) {
    std::vector<std::uint8_t> bytes;
    const Status enc = encode_stability_report(row, &bytes);
    if (!enc.ok) {
        return enc;
    }
    return write_bytes_with_info(path, bytes, out_info);
}

Status read_stability_report_file(const std::filesystem::path& path, StabilityReportRow* row) {
    return read_single_fixed_record(path, row, decode_stability_report);
}

Status write_top_assignments_file(const std::filesystem::path& path, const std::vector<TopAssignmentRow>& rows) {
    return write_top_assignments_file_with_info(path, rows, nullptr);
}

Status write_top_assignments_file_with_info(
    const std::filesystem::path& path,
    const std::vector<TopAssignmentRow>& rows,
    WriteArtifactInfo* out_info) {
    return write_rows_record_with_info(path, rows, encode_top_assignments, out_info);
}

Status read_top_assignments_file(const std::filesystem::path& path, std::vector<TopAssignmentRow>* rows) {
    return read_rows_record(path, rows, decode_top_assignments);
}

Status write_top_centroids_file(const std::filesystem::path& path, const std::vector<TopCentroidRow>& rows) {
    return write_top_centroids_file_with_info(path, rows, nullptr);
}

Status write_top_centroids_file_with_info(
    const std::filesystem::path& path,
    const std::vector<TopCentroidRow>& rows,
    WriteArtifactInfo* out_info) {
    return write_rows_record_with_info(path, rows, encode_top_centroids, out_info);
}

Status read_top_centroids_file(const std::filesystem::path& path, std::vector<TopCentroidRow>* rows) {
    return read_rows_record(path, rows, decode_top_centroids);
}

Status write_cluster_manifest_file(const std::filesystem::path& path, const std::vector<std::uint8_t>& payload) {
    return write_cluster_manifest_file_with_info(path, payload, nullptr);
}

Status write_cluster_manifest_file_with_info(
    const std::filesystem::path& path,
    const std::vector<std::uint8_t>& payload,
    WriteArtifactInfo* out_info) {
    CommonHeader header{};
    header.schema_version = 1U;
    header.record_type = 0x0F01U;
    header.record_count = 1U;
    std::vector<std::uint8_t> bytes;
    const Status enc = encode_header_plus_payload(header, payload, &bytes);
    if (!enc.ok) {
        return enc;
    }
    return write_bytes_with_info(path, bytes, out_info);
}

Status read_cluster_manifest_file(
    const std::filesystem::path& path,
    CommonHeader* header,
    std::vector<std::uint8_t>* payload) {
    std::vector<std::uint8_t> bytes;
    const Status rd = read_file_bytes(path, &bytes);
    if (!rd.ok) {
        return rd;
    }
    return decode_header_plus_payload(bytes, header, payload);
}

Status write_mid_assignments_file(const std::filesystem::path& path, const std::vector<MidAssignmentRow>& rows) {
    return write_mid_assignments_file_with_info(path, rows, nullptr);
}

Status write_mid_assignments_file_with_info(
    const std::filesystem::path& path,
    const std::vector<MidAssignmentRow>& rows,
    WriteArtifactInfo* out_info) {
    return write_rows_record_with_info(path, rows, encode_mid_assignments, out_info);
}

Status read_mid_assignments_file(const std::filesystem::path& path, std::vector<MidAssignmentRow>* rows) {
    return read_rows_record(path, rows, decode_mid_assignments);
}

Status write_final_assignments_file(const std::filesystem::path& path, const std::vector<FinalAssignmentRow>& rows) {
    return write_final_assignments_file_with_info(path, rows, nullptr);
}

Status write_final_assignments_file_with_info(
    const std::filesystem::path& path,
    const std::vector<FinalAssignmentRow>& rows,
    WriteArtifactInfo* out_info) {
    return write_rows_record_with_info(path, rows, encode_final_assignments, out_info);
}

Status read_final_assignments_file(const std::filesystem::path& path, std::vector<FinalAssignmentRow>* rows) {
    return read_rows_record(path, rows, decode_final_assignments);
}

Status write_k_search_bounds_batch_file(
    const std::filesystem::path& path,
    const std::vector<KSearchBoundsBatchRow>& rows) {
    return write_k_search_bounds_batch_file_with_info(path, rows, nullptr);
}

Status write_k_search_bounds_batch_file_with_info(
    const std::filesystem::path& path,
    const std::vector<KSearchBoundsBatchRow>& rows,
    WriteArtifactInfo* out_info) {
    return write_rows_record_with_info(path, rows, encode_k_search_bounds_batch, out_info);
}

Status read_k_search_bounds_batch_file(
    const std::filesystem::path& path,
    std::vector<KSearchBoundsBatchRow>* rows) {
    return read_rows_record(path, rows, decode_k_search_bounds_batch);
}

Status write_post_cluster_membership_file(
    const std::filesystem::path& path,
    const std::vector<PostClusterMembershipRow>& rows) {
    return write_post_cluster_membership_file_with_info(path, rows, nullptr);
}

Status write_post_cluster_membership_file_with_info(
    const std::filesystem::path& path,
    const std::vector<PostClusterMembershipRow>& rows,
    WriteArtifactInfo* out_info) {
    return write_rows_record_with_info(path, rows, encode_post_cluster_membership, out_info);
}

Status read_post_cluster_membership_file(
    const std::filesystem::path& path,
    std::vector<PostClusterMembershipRow>* rows) {
    return read_rows_record(path, rows, decode_post_cluster_membership);
}

Status write_embeddings_fp32_file(
    const std::filesystem::path& path,
    const std::vector<std::uint64_t>& ids,
    const std::vector<std::vector<float>>& vectors) {
    if (ids.size() != vectors.size()) {
        return Status::Error("embeddings_fp32.bin: ids/vectors size mismatch");
    }
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        if (vectors[i].size() != 1024U) {
            return Status::Error("embeddings_fp32.bin: vector dimension mismatch");
        }
        if (i > 0 && ids[i - 1] >= ids[i]) {
            return Status::Error("embeddings_fp32.bin: ids must be strictly ascending");
        }
    }
    const std::size_t record_size = kEmbeddingShardRecordSizeFp32;
    std::vector<std::uint8_t> bytes(kEmbeddingShardHeaderSize + ids.size() * record_size, 0U);
    store_le_u32(bytes.data() + 0U, kEmbeddingShardMagicFp32);
    store_le_u16(bytes.data() + 4U, 1U);
    store_le_u16(bytes.data() + 6U, static_cast<std::uint16_t>(EmbeddingShardValueType::FP32));
    store_le_u32(bytes.data() + 8U, static_cast<std::uint32_t>(record_size));
    store_le_u64(bytes.data() + 12U, static_cast<std::uint64_t>(ids.size()));
    store_le_u32(bytes.data() + 20U, 0U);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const std::size_t base = kEmbeddingShardHeaderSize + i * record_size;
        store_le_u64(bytes.data() + base, ids[i]);
        for (std::size_t d = 0; d < 1024U; ++d) {
            store_le_f32(bytes.data() + base + 8U + d * sizeof(float), vectors[i][d]);
        }
    }
    return write_atomic_bytes(path, bytes);
}

Status read_embeddings_fp32_file(
    const std::filesystem::path& path,
    std::vector<std::uint64_t>* ids,
    std::vector<std::vector<float>>* vectors) {
    const Status ids_guard = ensure_out(ids, "read_embeddings_fp32_file");
    if (!ids_guard.ok) {
        return ids_guard;
    }
    const Status vec_guard = ensure_out(vectors, "read_embeddings_fp32_file");
    if (!vec_guard.ok) {
        return vec_guard;
    }
    std::vector<std::uint8_t> bytes;
    const Status rd = read_file_bytes(path, &bytes);
    if (!rd.ok) {
        return rd;
    }
    if (bytes.size() < kEmbeddingShardHeaderSize) {
        return Status::Error("embeddings_fp32.bin: truncated header");
    }
    if (load_le_u32(bytes.data() + 0U) != kEmbeddingShardMagicFp32) {
        return Status::Error("embeddings_fp32.bin: bad magic");
    }
    if (load_le_u16(bytes.data() + 4U) != 1U) {
        return Status::Error("embeddings_fp32.bin: unsupported schema_version");
    }
    if (load_le_u16(bytes.data() + 6U) != static_cast<std::uint16_t>(EmbeddingShardValueType::FP32)) {
        return Status::Error("embeddings_fp32.bin: value_type mismatch");
    }
    const std::size_t record_size = load_le_u32(bytes.data() + 8U);
    const std::size_t record_count = static_cast<std::size_t>(load_le_u64(bytes.data() + 12U));
    if (record_size != kEmbeddingShardRecordSizeFp32) {
        return Status::Error("embeddings_fp32.bin: unexpected record_size");
    }
    if (bytes.size() != kEmbeddingShardHeaderSize + record_count * record_size) {
        return Status::Error("embeddings_fp32.bin: byte size mismatch");
    }
    ids->assign(record_count, 0U);
    vectors->assign(record_count, std::vector<float>(1024U, 0.0f));
    for (std::size_t i = 0; i < record_count; ++i) {
        const std::size_t base = kEmbeddingShardHeaderSize + i * record_size;
        (*ids)[i] = load_le_u64(bytes.data() + base);
        for (std::size_t d = 0; d < 1024U; ++d) {
            (*vectors)[i][d] = load_le_f32(bytes.data() + base + 8U + d * sizeof(float));
        }
    }
    return Status::Ok();
}

Status write_embeddings_fp16_file(
    const std::filesystem::path& path,
    const std::vector<std::uint64_t>& ids,
    const std::vector<std::vector<float>>& vectors) {
    if (ids.size() != vectors.size()) {
        return Status::Error("embeddings_fp16.bin: ids/vectors size mismatch");
    }
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        if (vectors[i].size() != 1024U) {
            return Status::Error("embeddings_fp16.bin: vector dimension mismatch");
        }
        if (i > 0 && ids[i - 1] >= ids[i]) {
            return Status::Error("embeddings_fp16.bin: ids must be strictly ascending");
        }
    }
    const std::size_t record_size = kEmbeddingShardRecordSizeFp16;
    std::vector<std::uint8_t> bytes(kEmbeddingShardHeaderSize + ids.size() * record_size, 0U);
    store_le_u32(bytes.data() + 0U, kEmbeddingShardMagicFp16);
    store_le_u16(bytes.data() + 4U, 1U);
    store_le_u16(bytes.data() + 6U, static_cast<std::uint16_t>(EmbeddingShardValueType::FP16));
    store_le_u32(bytes.data() + 8U, static_cast<std::uint32_t>(record_size));
    store_le_u64(bytes.data() + 12U, static_cast<std::uint64_t>(ids.size()));
    store_le_u32(bytes.data() + 20U, 0U);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const std::size_t base = kEmbeddingShardHeaderSize + i * record_size;
        store_le_u64(bytes.data() + base, ids[i]);
        for (std::size_t d = 0; d < 1024U; ++d) {
            store_le_u16(bytes.data() + base + 8U + d * sizeof(std::uint16_t), fp32_to_fp16_bits(vectors[i][d]));
        }
    }
    return write_atomic_bytes(path, bytes);
}

Status read_embeddings_fp16_file(
    const std::filesystem::path& path,
    std::vector<std::uint64_t>* ids,
    std::vector<std::vector<float>>* vectors) {
    const Status ids_guard = ensure_out(ids, "read_embeddings_fp16_file");
    if (!ids_guard.ok) {
        return ids_guard;
    }
    const Status vec_guard = ensure_out(vectors, "read_embeddings_fp16_file");
    if (!vec_guard.ok) {
        return vec_guard;
    }
    std::vector<std::uint8_t> bytes;
    const Status rd = read_file_bytes(path, &bytes);
    if (!rd.ok) {
        return rd;
    }
    if (bytes.size() < kEmbeddingShardHeaderSize) {
        return Status::Error("embeddings_fp16.bin: truncated header");
    }
    if (load_le_u32(bytes.data() + 0U) != kEmbeddingShardMagicFp16) {
        return Status::Error("embeddings_fp16.bin: bad magic");
    }
    if (load_le_u16(bytes.data() + 4U) != 1U) {
        return Status::Error("embeddings_fp16.bin: unsupported schema_version");
    }
    if (load_le_u16(bytes.data() + 6U) != static_cast<std::uint16_t>(EmbeddingShardValueType::FP16)) {
        return Status::Error("embeddings_fp16.bin: value_type mismatch");
    }
    const std::size_t record_size = load_le_u32(bytes.data() + 8U);
    const std::size_t record_count = static_cast<std::size_t>(load_le_u64(bytes.data() + 12U));
    if (record_size != kEmbeddingShardRecordSizeFp16) {
        return Status::Error("embeddings_fp16.bin: unexpected record_size");
    }
    if (bytes.size() != kEmbeddingShardHeaderSize + record_count * record_size) {
        return Status::Error("embeddings_fp16.bin: byte size mismatch");
    }
    ids->assign(record_count, 0U);
    vectors->assign(record_count, std::vector<float>(1024U, 0.0f));
    for (std::size_t i = 0; i < record_count; ++i) {
        const std::size_t base = kEmbeddingShardHeaderSize + i * record_size;
        (*ids)[i] = load_le_u64(bytes.data() + base);
        for (std::size_t d = 0; d < 1024U; ++d) {
            const std::uint16_t bits = load_le_u16(bytes.data() + base + 8U + d * sizeof(std::uint16_t));
            (*vectors)[i][d] = fp16_bits_to_fp32(bits);
        }
    }
    return Status::Ok();
}

Status write_embeddings_int8_sym_file(
    const std::filesystem::path& path,
    const std::vector<std::uint64_t>& ids,
    const std::vector<std::vector<float>>& vectors) {
    if (ids.size() != vectors.size()) {
        return Status::Error("embeddings_int8_sym.bin: ids/vectors size mismatch");
    }
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        if (vectors[i].size() != 1024U) {
            return Status::Error("embeddings_int8_sym.bin: vector dimension mismatch");
        }
        if (i > 0 && ids[i - 1] >= ids[i]) {
            return Status::Error("embeddings_int8_sym.bin: ids must be strictly ascending");
        }
    }
    const std::size_t record_size = kEmbeddingShardRecordSizeInt8Sym;
    std::vector<std::uint8_t> bytes(kEmbeddingShardHeaderSize + ids.size() * record_size, 0U);
    store_le_u32(bytes.data() + 0U, kEmbeddingShardMagicInt8Sym);
    store_le_u16(bytes.data() + 4U, 1U);
    store_le_u16(bytes.data() + 6U, static_cast<std::uint16_t>(EmbeddingShardValueType::INT8Sym));
    store_le_u32(bytes.data() + 8U, static_cast<std::uint32_t>(record_size));
    store_le_u64(bytes.data() + 12U, static_cast<std::uint64_t>(ids.size()));
    store_le_u32(bytes.data() + 20U, 0U);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const std::size_t base = kEmbeddingShardHeaderSize + i * record_size;
        store_le_u64(bytes.data() + base, ids[i]);
        float max_abs = 0.0f;
        for (float v : vectors[i]) {
            max_abs = std::max(max_abs, std::fabs(v));
        }
        const float scale = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
        store_le_f32(bytes.data() + base + 8U, scale);
        for (std::size_t d = 0; d < 1024U; ++d) {
            const float q = vectors[i][d] / scale;
            const int rounded = static_cast<int>(std::nearbyint(q));
            const int clamped = std::max(-127, std::min(127, rounded));
            bytes[base + 12U + d] = static_cast<std::uint8_t>(static_cast<std::int8_t>(clamped));
        }
    }
    return write_atomic_bytes(path, bytes);
}

Status read_embeddings_int8_sym_file(
    const std::filesystem::path& path,
    std::vector<std::uint64_t>* ids,
    std::vector<std::vector<float>>* vectors) {
    const Status ids_guard = ensure_out(ids, "read_embeddings_int8_sym_file");
    if (!ids_guard.ok) {
        return ids_guard;
    }
    const Status vec_guard = ensure_out(vectors, "read_embeddings_int8_sym_file");
    if (!vec_guard.ok) {
        return vec_guard;
    }
    std::vector<std::uint8_t> bytes;
    const Status rd = read_file_bytes(path, &bytes);
    if (!rd.ok) {
        return rd;
    }
    if (bytes.size() < kEmbeddingShardHeaderSize) {
        return Status::Error("embeddings_int8_sym.bin: truncated header");
    }
    if (load_le_u32(bytes.data() + 0U) != kEmbeddingShardMagicInt8Sym) {
        return Status::Error("embeddings_int8_sym.bin: bad magic");
    }
    if (load_le_u16(bytes.data() + 4U) != 1U) {
        return Status::Error("embeddings_int8_sym.bin: unsupported schema_version");
    }
    if (load_le_u16(bytes.data() + 6U) != static_cast<std::uint16_t>(EmbeddingShardValueType::INT8Sym)) {
        return Status::Error("embeddings_int8_sym.bin: value_type mismatch");
    }
    const std::size_t record_size = load_le_u32(bytes.data() + 8U);
    const std::size_t record_count = static_cast<std::size_t>(load_le_u64(bytes.data() + 12U));
    if (record_size != kEmbeddingShardRecordSizeInt8Sym) {
        return Status::Error("embeddings_int8_sym.bin: unexpected record_size");
    }
    if (bytes.size() != kEmbeddingShardHeaderSize + record_count * record_size) {
        return Status::Error("embeddings_int8_sym.bin: byte size mismatch");
    }
    ids->assign(record_count, 0U);
    vectors->assign(record_count, std::vector<float>(1024U, 0.0f));
    for (std::size_t i = 0; i < record_count; ++i) {
        const std::size_t base = kEmbeddingShardHeaderSize + i * record_size;
        (*ids)[i] = load_le_u64(bytes.data() + base);
        const float scale = load_le_f32(bytes.data() + base + 8U);
        for (std::size_t d = 0; d < 1024U; ++d) {
            const std::int8_t q = static_cast<std::int8_t>(bytes[base + 12U + d]);
            (*vectors)[i][d] = static_cast<float>(q) * scale;
        }
    }
    return Status::Ok();
}

Status extract_embedding_ids_from_shard_file(
    const std::filesystem::path& path,
    std::vector<std::uint64_t>* ids) {
    const Status guard = ensure_out(ids, "extract_embedding_ids_from_shard_file");
    if (!guard.ok) {
        return guard;
    }
    std::vector<std::uint8_t> bytes;
    const Status rd = read_file_bytes(path, &bytes);
    if (!rd.ok) {
        return rd;
    }
    if (bytes.size() < kEmbeddingShardHeaderSize) {
        return Status::Error("embedding shard: truncated header");
    }
    const std::uint32_t magic = load_le_u32(bytes.data() + 0U);
    const std::size_t record_size = load_le_u32(bytes.data() + 8U);
    const std::size_t record_count = static_cast<std::size_t>(load_le_u64(bytes.data() + 12U));
    if (magic != kEmbeddingShardMagicFp32 &&
        magic != kEmbeddingShardMagicFp16 &&
        magic != kEmbeddingShardMagicInt8Sym) {
        return Status::Error("embedding shard: unknown magic");
    }
    if (bytes.size() != kEmbeddingShardHeaderSize + record_count * record_size) {
        return Status::Error("embedding shard: byte size mismatch");
    }
    if (record_size < sizeof(std::uint64_t)) {
        return Status::Error("embedding shard: invalid record size");
    }
    ids->assign(record_count, 0U);
    for (std::size_t i = 0; i < record_count; ++i) {
        const std::size_t base = kEmbeddingShardHeaderSize + i * record_size;
        (*ids)[i] = load_le_u64(bytes.data() + base);
    }
    return Status::Ok();
}

Status regenerate_precision_shards(
    const std::filesystem::path& fp32_path,
    const std::filesystem::path& fp16_path,
    const std::filesystem::path& int8_sym_path,
    bool regenerate_fp16,
    bool regenerate_int8_sym) {
    std::vector<std::uint64_t> ids;
    std::vector<std::vector<float>> vectors;
    const Status read_st = read_embeddings_fp32_file(fp32_path, &ids, &vectors);
    if (!read_st.ok) {
        return read_st;
    }
    if (regenerate_fp16) {
        const Status st = write_embeddings_fp16_file(fp16_path, ids, vectors);
        if (!st.ok) {
            return st;
        }
    }
    if (regenerate_int8_sym) {
        const Status st = write_embeddings_int8_sym_file(int8_sym_path, ids, vectors);
        if (!st.ok) {
            return st;
        }
    }
    return Status::Ok();
}

Status validate_id_estimate(const IdEstimateRow& row) {
    if (row.k_min == 0U || row.k_max == 0U) {
        return Status::Error("id_estimate.bin: k_min and k_max must be non-zero");
    }
    if (row.k_min > row.k_max) {
        return Status::Error("id_estimate.bin: k_min > k_max");
    }
    if (row.reserved != 0U) {
        return Status::Error("id_estimate.bin: reserved must be zero");
    }
    return Status::Ok();
}

Status validate_elbow_trace(const std::vector<ElbowTraceRow>& rows, std::optional<std::uint32_t> chosen_k) {
    bool found_chosen = !chosen_k.has_value();
    for (const auto& row : rows) {
        if (row.probe_phase != ProbePhase::Coarse && row.probe_phase != ProbePhase::Fine) {
            return Status::Error("elbow_trace.bin: invalid probe_phase");
        }
        if (row.reserved[0] != 0U || row.reserved[1] != 0U || row.reserved[2] != 0U) {
            return Status::Error("elbow_trace.bin: reserved must be zero");
        }
        if (chosen_k.has_value() && row.k_value == *chosen_k) {
            found_chosen = true;
        }
    }
    if (!found_chosen) {
        return Status::Error("elbow_trace.bin: chosen_k not found in probes");
    }
    return Status::Ok();
}

Status validate_stability_report(const StabilityReportRow& row) {
    if (row.reserved != 0U) {
        return Status::Error("stability_report.bin: reserved must be zero");
    }
    const auto code = static_cast<std::uint16_t>(row.status_code);
    if (code > static_cast<std::uint16_t>(StabilityStatusCode::Fail)) {
        return Status::Error("stability_report.bin: invalid status_code");
    }
    return Status::Ok();
}

Status validate_top_assignments(const std::vector<TopAssignmentRow>& rows) {
    return validate_sorted_unique_embedding_ids(rows, "assignments.bin(top)");
}

Status validate_top_centroids(const std::vector<TopCentroidRow>& rows) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].top_centroid_numeric_id != static_cast<std::uint32_t>(i)) {
            return Status::Error("centroids.bin(top): centroid IDs must be dense ascending from zero");
        }
    }
    return Status::Ok();
}

Status validate_mid_assignments(const std::vector<MidAssignmentRow>& rows) {
    return validate_sorted_unique_embedding_ids(rows, "assignments.bin(mid)");
}

Status validate_final_assignments(const std::vector<FinalAssignmentRow>& rows) {
    return validate_sorted_unique_embedding_ids(rows, "assignments.bin(final)");
}

Status validate_k_search_bounds_batch(const std::vector<KSearchBoundsBatchRow>& rows) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        if (row.reserved != 0U) {
            return Status::Error("k_search_bounds_batch.bin: reserved must be zero");
        }
        if (row.k_min > row.chosen_k || row.chosen_k > row.k_max) {
            return Status::Error("k_search_bounds_batch.bin: k bounds invariant failed");
        }
        if (row.stage_level != StageLevel::Top &&
            row.stage_level != StageLevel::Mid &&
            row.stage_level != StageLevel::Lower) {
            return Status::Error("k_search_bounds_batch.bin: invalid stage_level");
        }
        if (row.stage_level == StageLevel::Top || row.stage_level == StageLevel::Mid) {
            if (row.gate_decision != GateDecision::NotApplicable) {
                return Status::Error("k_search_bounds_batch.bin: top/mid must have gate_decision=0");
            }
        } else {
            if (row.gate_decision != GateDecision::Continue &&
                row.gate_decision != GateDecision::Stop) {
                return Status::Error("k_search_bounds_batch.bin: lower must have gate_decision 1 or 2");
            }
        }
        if (i > 0) {
            const auto prev_stage = static_cast<std::uint8_t>(rows[i - 1].stage_level);
            const auto cur_stage = static_cast<std::uint8_t>(row.stage_level);
            if (prev_stage > cur_stage) {
                return Status::Error("k_search_bounds_batch.bin: not sorted by stage_level");
            }
            if (prev_stage == cur_stage &&
                rows[i - 1].source_numeric_id > row.source_numeric_id) {
                return Status::Error("k_search_bounds_batch.bin: not sorted by source_numeric_id");
            }
        }
    }
    return Status::Ok();
}

Status validate_post_cluster_membership(
    const std::vector<PostClusterMembershipRow>& rows,
    bool allow_lower_sentinel) {
    const Status sorted = validate_sorted_unique_embedding_ids(rows, "post_cluster_membership.bin");
    if (!sorted.ok) {
        return sorted;
    }
    if (!allow_lower_sentinel) {
        for (const auto& row : rows) {
            if (row.lower_centroid_numeric_id == std::numeric_limits<std::uint32_t>::max()) {
                return Status::Error("post_cluster_membership.bin: lower sentinel not allowed");
            }
        }
    }
    return Status::Ok();
}

PrecisionAlignmentResult evaluate_precision_id_alignment(
    const std::vector<std::uint64_t>& fp32_ids,
    const std::optional<std::vector<std::uint64_t>>& fp16_ids,
    const std::vector<std::vector<std::uint64_t>>& int8_variants) {
    PrecisionAlignmentResult out{};
    out.pass = false;

    auto check_sorted_unique = [&](const std::vector<std::uint64_t>& ids, const std::string& name) -> bool {
        for (std::size_t i = 1; i < ids.size(); ++i) {
            if (ids[i - 1] >= ids[i]) {
                out.reason = name + ": ids must be strictly ascending";
                out.mismatch_count = 1;
                return false;
            }
        }
        return true;
    };

    if (!check_sorted_unique(fp32_ids, "fp32")) {
        return out;
    }
    if (fp16_ids.has_value() && !check_sorted_unique(*fp16_ids, "fp16")) {
        return out;
    }
    for (std::size_t i = 0; i < int8_variants.size(); ++i) {
        if (!check_sorted_unique(int8_variants[i], "int8[" + std::to_string(i) + "]")) {
            return out;
        }
    }

    auto compare_with_fp32 = [&](const std::vector<std::uint64_t>& other, const std::string& name) -> bool {
        if (other.size() != fp32_ids.size()) {
            out.reason = name + ": cardinality mismatch";
            out.mismatch_count = (other.size() > fp32_ids.size()) ?
                                     (other.size() - fp32_ids.size()) :
                                     (fp32_ids.size() - other.size());
            return false;
        }
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < fp32_ids.size(); ++i) {
            if (fp32_ids[i] != other[i]) {
                ++mismatches;
            }
        }
        if (mismatches > 0) {
            out.reason = name + ": membership mismatch";
            out.mismatch_count = mismatches;
            return false;
        }
        return true;
    };

    if (fp16_ids.has_value() && !compare_with_fp32(*fp16_ids, "fp16")) {
        return out;
    }
    for (std::size_t i = 0; i < int8_variants.size(); ++i) {
        if (!compare_with_fp32(int8_variants[i], "int8[" + std::to_string(i) + "]")) {
            return out;
        }
    }

    out.pass = true;
    out.mismatch_count = 0;
    out.reason.clear();
    return out;
}

Status validate_precision_id_alignment(
    const std::vector<std::uint64_t>& fp32_ids,
    const std::optional<std::vector<std::uint64_t>>& fp16_ids,
    const std::vector<std::vector<std::uint64_t>>& int8_variants) {
    const PrecisionAlignmentResult r =
        evaluate_precision_id_alignment(fp32_ids, fp16_ids, int8_variants);
    if (!r.pass) {
        return Status::Error("precision alignment failed: " + r.reason);
    }
    return Status::Ok();
}

}  // namespace vector_db_v3::codec
