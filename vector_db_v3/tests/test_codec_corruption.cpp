#include <iostream>
#include <limits>
#include <optional>
#include <vector>

#include "vector_db_v3/codec/artifacts.hpp"

using namespace vector_db_v3::codec;

namespace {

bool expect_fail(const vector_db_v3::Status& s, const char* msg) {
    if (s.ok) {
        std::cerr << "FAIL: expected failure: " << msg << "\n";
        return false;
    }
    return true;
}

bool expect_ok(const vector_db_v3::Status& s, const char* msg) {
    if (!s.ok) {
        std::cerr << "FAIL: expected ok: " << msg << " (" << s.message << ")\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    bool ok = true;

    {
        std::vector<std::uint8_t> bad = {0U, 1U, 2U};
        std::vector<TopAssignmentRow> out;
        ok &= expect_fail(decode_top_assignments(bad, &out), "bad top assignment length");
    }

    {
        std::vector<ElbowTraceRow> rows = {
            {8U, 1.0f, static_cast<ProbePhase>(9U), {0U, 0U, 0U}},
        };
        ok &= expect_fail(validate_elbow_trace(rows), "invalid probe phase");
    }

    {
        std::vector<ElbowTraceRow> rows = {
            {8U, 1.0f, ProbePhase::Coarse, {1U, 0U, 0U}},
        };
        ok &= expect_fail(validate_elbow_trace(rows), "non-zero elbow reserved");
    }

    {
        IdEstimateRow row{10U, 1U, 1U, 0U};
        ok &= expect_fail(validate_id_estimate(row), "k_min > k_max");
    }

    {
        StabilityReportRow row{};
        row.status_code = static_cast<StabilityStatusCode>(9U);
        ok &= expect_fail(validate_stability_report(row), "invalid stability enum");
    }

    {
        std::vector<TopAssignmentRow> rows = {{10U, 1U}, {10U, 2U}};
        ok &= expect_fail(validate_top_assignments(rows), "duplicate embedding id");
    }

    {
        std::vector<KSearchBoundsBatchRow> rows = {
            {StageLevel::Top, GateDecision::Continue, 0U, 1U, 2U, 4U, 3U, 9U},
        };
        ok &= expect_fail(validate_k_search_bounds_batch(rows), "invalid top gate");
    }

    {
        std::vector<KSearchBoundsBatchRow> rows = {
            {StageLevel::Lower, GateDecision::NotApplicable, 0U, 1U, 2U, 4U, 3U, 9U},
        };
        ok &= expect_fail(validate_k_search_bounds_batch(rows), "invalid lower gate");
    }

    {
        CommonHeader hdr{};
        hdr.schema_version = 1U;
        hdr.record_type = 1U;
        hdr.record_count = 1U;
        std::vector<std::uint8_t> payload = {1U, 2U, 3U};
        std::vector<std::uint8_t> bytes;
        ok &= expect_ok(encode_header_plus_payload(hdr, payload, &bytes), "encode header+payload");
        bytes.back() ^= 0xFFU;
        CommonHeader out_hdr{};
        std::vector<std::uint8_t> out_payload;
        ok &= expect_fail(
            decode_header_plus_payload(bytes, &out_hdr, &out_payload),
            "checksum mismatch should fail");
    }

    {
        const std::vector<std::uint64_t> fp32 = {1U, 2U, 3U};
        const std::optional<std::vector<std::uint64_t>> fp16 = std::vector<std::uint64_t>{1U, 4U, 3U};
        const std::vector<std::vector<std::uint64_t>> int8 = {};
        const auto res = evaluate_precision_id_alignment(fp32, fp16, int8);
        if (res.pass || res.mismatch_count == 0U) {
            std::cerr << "FAIL: precision mismatch expected\n";
            ok = false;
        }
    }

    {
        std::vector<PostClusterMembershipRow> rows = {
            {2U, 1U, 1U, std::numeric_limits<std::uint32_t>::max(), 9U},
            {1U, 1U, 1U, 2U, 9U},
        };
        ok &= expect_fail(validate_post_cluster_membership(rows, false), "unsorted membership and sentinel forbidden");
    }

    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_codec_corruption_tests: PASS\n";
    return 0;
}
