#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "vector_db_v3/codec/artifacts.hpp"
#include "vector_db_v3/paths.hpp"
#include "vector_db_v3/vector_store.hpp"

namespace fs = std::filesystem;

namespace {

bool expect(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        return false;
    }
    return true;
}

std::vector<float> make_vec(float base) {
    std::vector<float> v(vector_db_v3::kVectorDim, 0.0f);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = base + static_cast<float>((i % 7U) * 0.01f);
    }
    return v;
}

}  // namespace

int main() {
    const fs::path data_dir = fs::temp_directory_path() / "vectordb_v3_top_layer_artifacts_test";
    std::error_code ec;
    fs::remove_all(data_dir, ec);

    bool ok = true;
    vector_db_v3::VectorStore store(data_dir.string());
    ok &= expect(store.init().ok, "init");
    ok &= expect(store.open().ok, "open");
    for (std::uint64_t id = 1; id <= 24; ++id) {
        ok &= expect(store.insert(id, make_vec(static_cast<float>(id) * 0.1f)).ok, "insert");
    }

    ok &= expect(store.build_top_clusters(123U).ok, "build_top_clusters");

    const fs::path id_estimate = vector_db_v3::paths::id_estimate_bin(data_dir);
    const fs::path elbow_trace = vector_db_v3::paths::elbow_trace_bin(data_dir);
    const fs::path centroids = vector_db_v3::paths::centroids_bin(data_dir);
    const fs::path assignments = vector_db_v3::paths::top_assignments_bin(data_dir);
    const fs::path stability = vector_db_v3::paths::stability_report_bin(data_dir);
    const fs::path manifest = vector_db_v3::paths::cluster_manifest_bin(data_dir);

    ok &= expect(fs::exists(id_estimate), "id_estimate.bin exists");
    ok &= expect(fs::exists(elbow_trace), "elbow_trace.bin exists");
    ok &= expect(fs::exists(centroids), "centroids.bin exists");
    ok &= expect(fs::exists(assignments), "assignments.bin exists");
    ok &= expect(fs::exists(stability), "stability_report.bin exists");
    ok &= expect(fs::exists(manifest), "cluster_manifest.bin exists");

    vector_db_v3::codec::IdEstimateRow estimate{};
    ok &= expect(vector_db_v3::codec::read_id_estimate_file(id_estimate, &estimate).ok, "read id_estimate");
    ok &= expect(estimate.k_min > 0U && estimate.k_min <= estimate.k_max, "id_estimate bounds");

    std::vector<vector_db_v3::codec::ElbowTraceRow> elbow_rows;
    ok &= expect(vector_db_v3::codec::read_elbow_trace_file(elbow_trace, &elbow_rows).ok, "read elbow_trace");
    ok &= expect(!elbow_rows.empty(), "elbow_trace non-empty");

    std::vector<vector_db_v3::codec::TopCentroidRow> centroid_rows;
    ok &= expect(vector_db_v3::codec::read_top_centroids_file(centroids, &centroid_rows).ok, "read centroids");
    ok &= expect(!centroid_rows.empty(), "centroid rows non-empty");

    std::vector<vector_db_v3::codec::TopAssignmentRow> assignment_rows;
    ok &= expect(vector_db_v3::codec::read_top_assignments_file(assignments, &assignment_rows).ok, "read assignments");
    ok &= expect(assignment_rows.size() == 24U, "assignments size matches inserted rows");
    for (const auto& row : assignment_rows) {
        ok &= expect(row.top_centroid_numeric_id < centroid_rows.size(), "assignment id in centroid range");
    }

    vector_db_v3::codec::StabilityReportRow stability_row{};
    ok &= expect(vector_db_v3::codec::read_stability_report_file(stability, &stability_row).ok, "read stability_report");
    ok &= expect(stability_row.status_code == vector_db_v3::codec::StabilityStatusCode::Ok, "stability status ok");

    vector_db_v3::codec::CommonHeader manifest_header{};
    std::vector<std::uint8_t> manifest_payload;
    ok &= expect(
        vector_db_v3::codec::read_cluster_manifest_file(manifest, &manifest_header, &manifest_payload).ok,
        "read cluster_manifest");
    ok &= expect(manifest_header.schema_version == 1U, "manifest schema version");
    const std::string payload_str(manifest_payload.begin(), manifest_payload.end());
    ok &= expect(payload_str.find("id_estimate.bin") != std::string::npos, "manifest includes id_estimate");
    ok &= expect(payload_str.find("elbow_trace.bin") != std::string::npos, "manifest includes elbow_trace");
    ok &= expect(payload_str.find("centroids.bin") != std::string::npos, "manifest includes centroids");
    ok &= expect(payload_str.find("assignments.bin") != std::string::npos, "manifest includes assignments");
    ok &= expect(payload_str.find("stability_report.bin") != std::string::npos, "manifest includes stability_report");

    ok &= expect(store.close().ok, "close");
    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_top_layer_artifacts_tests: PASS\n";
    return 0;
}
