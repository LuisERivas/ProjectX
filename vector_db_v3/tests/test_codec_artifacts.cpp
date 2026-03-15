#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

#include "vector_db_v3/codec/artifacts.hpp"
#include "vector_db_v3/codec/checksum.hpp"

namespace fs = std::filesystem;
using namespace vector_db_v3::codec;

namespace {

bool expect(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    bool ok = true;

    {
        IdEstimateRow in{4U, 16U, 1U, 0U};
        std::vector<std::uint8_t> bytes;
        IdEstimateRow out{};
        ok &= expect(encode_id_estimate(in, &bytes).ok, "encode id_estimate");
        ok &= expect(decode_id_estimate(bytes, &out).ok, "decode id_estimate");
        ok &= expect(out.k_min == 4U && out.k_max == 16U, "id_estimate roundtrip");
    }

    {
        std::vector<ElbowTraceRow> in = {
            {8U, 1.25f, ProbePhase::Coarse, {0U, 0U, 0U}},
            {12U, 0.95f, ProbePhase::Fine, {0U, 0U, 0U}},
        };
        std::vector<std::uint8_t> bytes;
        std::vector<ElbowTraceRow> out;
        ok &= expect(encode_elbow_trace(in, &bytes).ok, "encode elbow");
        ok &= expect(decode_elbow_trace(bytes, &out).ok, "decode elbow");
        ok &= expect(out.size() == 2U, "elbow row count");
        ok &= expect(validate_elbow_trace(out, 12U).ok, "elbow chosen_k present");
    }

    {
        StabilityReportRow in{};
        in.status_code = StabilityStatusCode::Ok;
        in.mean_nmi = 0.9f;
        in.std_nmi = 0.1f;
        in.mean_jaccard = 0.8f;
        in.mean_centroid_drift = 0.05f;
        std::vector<std::uint8_t> bytes;
        StabilityReportRow out{};
        ok &= expect(encode_stability_report(in, &bytes).ok, "encode stability");
        ok &= expect(decode_stability_report(bytes, &out).ok, "decode stability");
        ok &= expect(out.status_code == StabilityStatusCode::Ok, "stability status");
    }

    {
        std::vector<TopAssignmentRow> top = {{10U, 1U}, {11U, 1U}, {20U, 2U}};
        std::vector<std::uint8_t> bytes;
        std::vector<TopAssignmentRow> decoded;
        ok &= expect(encode_top_assignments(top, &bytes).ok, "encode top assignments");
        ok &= expect(decode_top_assignments(bytes, &decoded).ok, "decode top assignments");
        ok &= expect(validate_top_assignments(decoded).ok, "validate top assignments");
    }

    {
        std::vector<MidAssignmentRow> mid = {{10U, 5U, 1U}, {11U, 5U, 1U}, {20U, 6U, 2U}};
        std::vector<FinalAssignmentRow> fin = {{10U, 50U}, {11U, 50U}, {20U, 60U}};
        std::vector<std::uint8_t> b1;
        std::vector<std::uint8_t> b2;
        std::vector<MidAssignmentRow> d1;
        std::vector<FinalAssignmentRow> d2;
        ok &= expect(encode_mid_assignments(mid, &b1).ok, "encode mid");
        ok &= expect(encode_final_assignments(fin, &b2).ok, "encode final");
        ok &= expect(decode_mid_assignments(b1, &d1).ok, "decode mid");
        ok &= expect(decode_final_assignments(b2, &d2).ok, "decode final");
    }

    {
        std::vector<KSearchBoundsBatchRow> rows = {
            {StageLevel::Top, GateDecision::NotApplicable, 0U, 1U, 4U, 16U, 8U, 100U},
            {StageLevel::Mid, GateDecision::NotApplicable, 0U, 2U, 2U, 8U, 5U, 40U},
            {StageLevel::Lower, GateDecision::Stop, 0U, 9U, 2U, 4U, 3U, 10U},
        };
        std::vector<std::uint8_t> bytes;
        std::vector<KSearchBoundsBatchRow> decoded;
        ok &= expect(encode_k_search_bounds_batch(rows, &bytes).ok, "encode k_bounds");
        ok &= expect(decode_k_search_bounds_batch(bytes, &decoded).ok, "decode k_bounds");
        ok &= expect(validate_k_search_bounds_batch(decoded).ok, "validate k_bounds");
    }

    {
        std::vector<PostClusterMembershipRow> rows = {
            {10U, 1U, 11U, 111U, 1111U},
            {11U, 1U, 11U, std::numeric_limits<std::uint32_t>::max(), 1112U},
        };
        std::vector<std::uint8_t> bytes;
        std::vector<PostClusterMembershipRow> decoded;
        ok &= expect(encode_post_cluster_membership(rows, &bytes).ok, "encode membership");
        ok &= expect(decode_post_cluster_membership(bytes, &decoded).ok, "decode membership");
        ok &= expect(validate_post_cluster_membership(decoded, true).ok, "validate membership");
    }

    {
        CommonHeader hdr{};
        hdr.schema_version = 1U;
        hdr.record_type = 42U;
        hdr.record_count = 2U;
        const std::vector<std::uint8_t> payload = {1U, 2U, 3U, 4U};
        std::vector<std::uint8_t> bytes;
        CommonHeader out_hdr{};
        std::vector<std::uint8_t> out_payload;
        ok &= expect(encode_header_plus_payload(hdr, payload, &bytes).ok, "encode header+payload");
        ok &= expect(decode_header_plus_payload(bytes, &out_hdr, &out_payload).ok, "decode header+payload");
        ok &= expect(out_hdr.payload_bytes == payload.size(), "header payload bytes");
        ok &= expect(out_hdr.checksum_crc32 == crc32(payload), "header crc32");
        ok &= expect(out_payload == payload, "payload roundtrip");
    }

    {
        const std::vector<std::uint64_t> fp32 = {1U, 2U, 3U};
        const std::optional<std::vector<std::uint64_t>> fp16 = std::vector<std::uint64_t>{1U, 2U, 3U};
        const std::vector<std::vector<std::uint64_t>> int8 = {{1U, 2U, 3U}, {1U, 2U, 3U}};
        const auto res = evaluate_precision_id_alignment(fp32, fp16, int8);
        ok &= expect(res.pass, "precision alignment pass");
    }

    {
        const fs::path root = fs::temp_directory_path() / "vectordb_v3_codec_artifacts";
        std::error_code ec;
        fs::remove_all(root, ec);
        IdEstimateRow row{2U, 4U, 1U, 0U};
        IdEstimateRow loaded{};
        ok &= expect(write_id_estimate_file(root / "id_estimate.bin", row).ok, "write id_estimate file");
        ok &= expect(read_id_estimate_file(root / "id_estimate.bin", &loaded).ok, "read id_estimate file");
        ok &= expect(loaded.k_max == 4U, "id_estimate file roundtrip");
    }

    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_codec_artifacts_tests: PASS\n";
    return 0;
}
