#include <filesystem>
#include <fstream>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "vector_db/vector_store.hpp"

namespace fs = std::filesystem;

namespace {

bool expect(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << "\n";
        return false;
    }
    return true;
}

std::vector<float> make_vec(float base) {
    std::vector<float> v(vector_db::kVectorDim, 0.0f);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = base + static_cast<float>(i) * 0.001f;
    }
    return v;
}

std::vector<float> make_unit_vec(std::size_t hot_idx, float strength, std::size_t variant) {
    std::vector<float> v(vector_db::kVectorDim, 0.0f);
    v[hot_idx % vector_db::kVectorDim] = strength;
    v[(hot_idx + 1) % vector_db::kVectorDim] = 1.0f - strength;
    // Add tiny deterministic variation so vectors are not exact duplicates.
    const std::size_t j1 = (hot_idx + 7 + variant) % vector_db::kVectorDim;
    const std::size_t j2 = (hot_idx + 31 + (variant * 3)) % vector_db::kVectorDim;
    v[j1] += 0.0005f * static_cast<float>((variant % 11) + 1);
    v[j2] += 0.0003f * static_cast<float>(((variant + 5) % 13) + 1);
    float norm_sq = 0.0f;
    for (float x : v) {
        norm_sq += x * x;
    }
    const float norm = std::sqrt(norm_sq);
    for (float& x : v) {
        x /= norm;
    }
    return v;
}

std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream os;
    os << in.rdbuf();
    return os.str();
}

std::size_t count_non_empty_lines(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::string line;
    std::size_t count = 0;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            count += 1;
        }
    }
    return count;
}

std::vector<std::uint64_t> parse_lsns(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::vector<std::uint64_t> out;
    std::string line;
    while (std::getline(in, line)) {
        const auto key = line.find("\"lsn\":");
        if (key == std::string::npos) {
            continue;
        }
        const auto n0 = line.find_first_of("0123456789", key);
        const auto n1 = line.find_first_not_of("0123456789", n0);
        out.push_back(static_cast<std::uint64_t>(std::stoull(line.substr(n0, n1 - n0))));
    }
    return out;
}

std::size_t count_substr(const std::string& text, const std::string& needle) {
    std::size_t count = 0;
    std::size_t pos = 0;
    while (true) {
        pos = text.find(needle, pos);
        if (pos == std::string::npos) {
            break;
        }
        ++count;
        pos += needle.size();
    }
    return count;
}

}  // namespace

int main() {
    const fs::path test_dir = fs::path("vector_db_test_data");
    std::error_code ec;
    fs::remove_all(test_dir, ec);

    const auto v1 = make_vec(1.0f);
    const auto v2 = make_vec(2.0f);

    // A) WAL append correctness + baseline CRUD path.
    vector_db::VectorStore store(test_dir.string());
    if (!expect(store.init().ok, "init succeeds")) {
        return 1;
    }
    if (!expect(store.open().ok, "open succeeds")) {
        return 1;
    }
    if (!expect(store.insert(42, v1, "{\"tag\":\"first\"}").ok, "insert #1")) {
        return 1;
    }
    if (!expect(store.update_metadata(42, "{\"tag\":\"updated\",\"k\":\"v\"}").ok, "metadata update")) {
        return 1;
    }
    if (!expect(store.remove(42).ok, "delete existing id")) {
        return 1;
    }

    const fs::path wal = test_dir / "wal.log";
    if (!expect(fs::exists(wal), "wal exists")) {
        return 1;
    }
    if (!expect(count_non_empty_lines(wal) == 3, "wal has 3 records")) {
        return 1;
    }
    const auto wal_text = slurp(wal);
    if (!expect(wal_text.find("\"op\":\"INSERT\"") != std::string::npos, "wal contains INSERT")) {
        return 1;
    }
    if (!expect(wal_text.find("\"op\":\"UPDATE_META\"") != std::string::npos, "wal contains UPDATE_META")) {
        return 1;
    }
    if (!expect(wal_text.find("\"op\":\"DELETE\"") != std::string::npos, "wal contains DELETE")) {
        return 1;
    }
    const auto lsns = parse_lsns(wal);
    if (!expect(lsns.size() == 3, "lsn parsed for every wal record")) {
        return 1;
    }
    if (!expect(lsns[0] < lsns[1] && lsns[1] < lsns[2], "lsn monotonic increase")) {
        return 1;
    }

    // Baseline state checks.
    const auto rec42 = store.get(42);
    if (!expect(rec42.has_value() && rec42->deleted, "record tombstoned after delete")) {
        return 1;
    }
    if (!expect(store.insert(7, v2, "{\"tag\":\"second\"}").ok, "insert #2")) {
        return 1;
    }
    if (!expect(!store.insert(7, v2, "{\"tag\":\"dup\"}").ok, "duplicate id rejected")) {
        return 1;
    }
    const auto st = store.stats();
    if (!expect(st.total_rows == 2, "stats total rows")) {
        return 1;
    }
    if (!expect(st.live_rows == 1, "stats live rows")) {
        return 1;
    }
    if (!expect(st.tombstone_rows == 1, "stats tombstone rows")) {
        return 1;
    }
    if (!expect(fs::exists(test_dir / "manifest.json"), "manifest file written")) {
        return 1;
    }
    if (!expect(fs::exists(test_dir / "dirty_ranges.json"), "dirty ranges file written")) {
        return 1;
    }

    // B) Replay after simulated crash: no close, then reopen a fresh instance.
    vector_db::VectorStore crash_reopen(test_dir.string());
    if (!expect(crash_reopen.open().ok, "reopen after simulated crash succeeds")) {
        return 1;
    }
    const auto rec7 = crash_reopen.get(7);
    if (!expect(rec7.has_value() && !rec7->deleted, "reloaded live record after replay")) {
        return 1;
    }
    const auto rec42_after = crash_reopen.get(42);
    if (!expect(rec42_after.has_value() && rec42_after->deleted, "reloaded tombstoned record after replay")) {
        return 1;
    }

    // C) Idempotent repeated replay: second open should not change counts.
    const auto st_once = crash_reopen.stats();
    vector_db::VectorStore second_reopen(test_dir.string());
    if (!expect(second_reopen.open().ok, "second reopen succeeds")) {
        return 1;
    }
    const auto st_twice = second_reopen.stats();
    if (!expect(st_once.total_rows == st_twice.total_rows, "repeated open keeps total rows")) {
        return 1;
    }
    if (!expect(st_once.tombstone_rows == st_twice.tombstone_rows, "repeated open keeps tombstones")) {
        return 1;
    }

    // D) Checkpoint truncation behavior.
    if (!expect(second_reopen.checkpoint().ok, "checkpoint succeeds")) {
        return 1;
    }
    const auto wal_after_checkpoint = second_reopen.wal_stats();
    if (!expect(wal_after_checkpoint.wal_entries == 0, "wal truncated after checkpoint")) {
        return 1;
    }
    if (!expect(wal_after_checkpoint.checkpoint_lsn >= lsns.back(), "checkpoint_lsn advanced")) {
        return 1;
    }

    // E) Corrupted trailing WAL line tolerance.
    if (!expect(second_reopen.insert(99, v1, "{\"tag\":\"tail\"}").ok, "insert before corrupt tail")) {
        return 1;
    }
    {
        std::ofstream out(test_dir / "wal.log", std::ios::binary | std::ios::app);
        out << "{\"lsn\":999999,\"op\":\"INSERT\"";  // malformed trailing line
    }
    vector_db::VectorStore malformed_tail_reopen(test_dir.string());
    if (!expect(malformed_tail_reopen.open().ok, "open succeeds with malformed trailing wal line")) {
        return 1;
    }
    const auto rec99 = malformed_tail_reopen.get(99);
    if (!expect(rec99.has_value() && !rec99->deleted, "valid wal records still replay with malformed tail")) {
        return 1;
    }

    // Phase 3: initial clustering build + reload.
    const fs::path cluster_dir = fs::path("vector_db_test_data_phase3");
    fs::remove_all(cluster_dir, ec);
    vector_db::VectorStore cluster_store(cluster_dir.string());
    if (!expect(cluster_store.init().ok, "phase3 init succeeds")) {
        return 1;
    }
    if (!expect(cluster_store.open().ok, "phase3 open succeeds")) {
        return 1;
    }
    for (std::uint64_t i = 0; i < 32; ++i) {
        const auto vec = make_unit_vec(static_cast<std::size_t>(i % 2 == 0 ? 10 : 200), 0.95f, static_cast<std::size_t>(i));
        if (!expect(cluster_store.insert(1000 + i, vec, "{\"kind\":\"cluster\"}").ok, "phase3 insert")) {
            return 1;
        }
    }
    if (!expect(cluster_store.build_initial_clusters(777).ok, "build initial clusters succeeds")) {
        return 1;
    }
    const auto cstats = cluster_store.cluster_stats();
    if (!expect(cstats.available, "cluster stats available")) {
        return 1;
    }
    if (!expect(cstats.version >= 1, "cluster version set")) {
        return 1;
    }
    if (!expect(cstats.chosen_k >= cstats.k_min && cstats.chosen_k <= cstats.k_max, "chosen_k in discovered range")) {
        return 1;
    }
    if (!expect(!cstats.gpu_backend.empty(), "cluster backend telemetry populated")) {
        return 1;
    }
    if (!expect(cstats.scoring_calls > 0, "cluster scoring call telemetry populated")) {
        return 1;
    }
    if (!expect(fs::exists(cluster_dir / "clusters" / "initial" / "cluster_manifest.json"), "cluster manifest exists")) {
        return 1;
    }
    if (!expect(fs::exists(cluster_dir / "clusters" / "initial" / ("v" + std::to_string(cstats.version)) / "id_estimate.json"), "id estimate artifact exists")) {
        return 1;
    }
    if (!expect(fs::exists(cluster_dir / "clusters" / "initial" / ("v" + std::to_string(cstats.version)) / "elbow_trace.json"), "elbow trace artifact exists")) {
        return 1;
    }
    if (!expect(fs::exists(cluster_dir / "clusters" / "initial" / ("v" + std::to_string(cstats.version)) / "stability_report.json"), "stability artifact exists")) {
        return 1;
    }
    const auto elbow_trace = slurp(cluster_dir / "clusters" / "initial" / ("v" + std::to_string(cstats.version)) / "elbow_trace.json");
    const std::size_t evaluated_k = count_substr(elbow_trace, "\"k\": ");
    if (!expect(evaluated_k >= 2, "elbow trace has multiple k evaluations")) {
        return 1;
    }
    if (!expect(evaluated_k <= 16, "elbow search remains bounded")) {
        return 1;
    }
    if (!expect(cluster_store.close().ok, "phase3 close succeeds")) {
        return 1;
    }
    vector_db::VectorStore cluster_reopen(cluster_dir.string());
    if (!expect(cluster_reopen.open().ok, "phase3 reopen succeeds")) {
        return 1;
    }
    const auto cstats_reopen = cluster_reopen.cluster_stats();
    if (!expect(cstats_reopen.available && cstats_reopen.version == cstats.version, "cluster stats stable across reopen")) {
        return 1;
    }
    if (!expect(cluster_reopen.build_second_level_clusters(888, cstats.version).ok, "build second-level clusters succeeds")) {
        return 1;
    }
    const fs::path second_level_doc =
        cluster_dir / "clusters" / "initial" / ("v" + std::to_string(cstats.version))
        / "second_level_clustering" / "SECOND_LEVEL_CLUSTERING.json";
    if (!expect(fs::exists(second_level_doc), "second-level clustering summary exists")) {
        return 1;
    }
    const auto second_level_text = slurp(second_level_doc);
    if (!expect(
            second_level_text.find("\"processed_centroids\"") != std::string::npos,
            "second-level processed count present")) {
        return 1;
    }
    if (!expect(
            second_level_text.find("\"total_parent_centroids\"") != std::string::npos,
            "second-level parent centroid count present")) {
        return 1;
    }

    std::cout << "[PASS] vectordb phase3 initial clustering tests\n";
    malformed_tail_reopen.close();
    cluster_reopen.close();
    fs::remove_all(test_dir, ec);
    fs::remove_all(cluster_dir, ec);
    return 0;
}

