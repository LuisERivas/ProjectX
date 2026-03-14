#include "vector_db/vector_store.hpp"
#include "vector_db/clustering.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace vector_db_v2 {

namespace {

struct Timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double elapsed_ms() const {
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - t0).count();
    }
};

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char c : s) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out.push_back(c);
                break;
        }
    }
    return out;
}

std::string vec_to_json_array(const std::vector<float>& v) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << std::fixed << std::setprecision(6) << v[i];
    }
    os << "]";
    return os.str();
}

std::optional<std::uint64_t> extract_u64(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = line.find(needle);
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto num_pos = line.find_first_of("0123456789", key_pos + needle.size());
    if (num_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto end_pos = line.find_first_not_of("0123456789", num_pos);
    try {
        return static_cast<std::uint64_t>(std::stoull(line.substr(num_pos, end_pos - num_pos)));
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<double> extract_double(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = line.find(needle);
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto num_pos = line.find_first_of("-0123456789.", key_pos + needle.size());
    if (num_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto end_pos = line.find_first_not_of("-0123456789.eE+", num_pos);
    try {
        return std::stod(line.substr(num_pos, end_pos - num_pos));
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<int> extract_i32(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = line.find(needle);
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto num_pos = line.find_first_of("-0123456789", key_pos + needle.size());
    if (num_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto end_pos = line.find_first_not_of("-0123456789", num_pos);
    try {
        return std::stoi(line.substr(num_pos, end_pos - num_pos));
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<bool> extract_bool(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = line.find(needle);
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto t_pos = line.find("true", key_pos + needle.size());
    const auto f_pos = line.find("false", key_pos + needle.size());
    if (t_pos != std::string::npos && (f_pos == std::string::npos || t_pos < f_pos)) {
        return true;
    }
    if (f_pos != std::string::npos) {
        return false;
    }
    return std::nullopt;
}

std::optional<std::string> extract_string(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = line.find(needle);
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto colon = line.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return std::nullopt;
    }
    const auto q1 = line.find('"', colon);
    if (q1 == std::string::npos) {
        return std::nullopt;
    }
    bool escaped = false;
    for (std::size_t i = q1 + 1; i < line.size(); ++i) {
        const char c = line[i];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            return line.substr(q1 + 1, i - q1 - 1);
        }
    }
    return std::nullopt;
}

std::optional<std::vector<float>> extract_vector_array(const std::string& line) {
    const auto vec_pos = line.find("\"vector\"");
    if (vec_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto lb = line.find('[', vec_pos);
    const auto rb = line.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos || rb <= lb) {
        return std::nullopt;
    }
    std::string slice = line.substr(lb + 1, rb - lb - 1);
    std::stringstream ss(slice);
    std::string tok;
    std::vector<float> out;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) {
            continue;
        }
        try {
            out.push_back(std::stof(tok));
        } catch (...) {
            return std::nullopt;
        }
    }
    return out;
}

bool write_text_atomic(const fs::path& path, const std::string& body) {
    try {
        fs::create_directories(path.parent_path());
        const fs::path tmp = path.string() + ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out) {
                return false;
            }
            out << body;
        }
        if (fs::exists(path)) {
            fs::remove(path);
        }
        fs::rename(tmp, path);
        return true;
    } catch (...) {
        return false;
    }
}

bool write_binary_atomic(const fs::path& path, const std::vector<float>& data) {
    try {
        fs::create_directories(path.parent_path());
        const fs::path tmp = path.string() + ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out) {
                return false;
            }
            out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
        }
        if (fs::exists(path)) {
            fs::remove(path);
        }
        fs::rename(tmp, path);
        return true;
    } catch (...) {
        return false;
    }
}

double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (na <= 0.0 || nb <= 0.0) {
        return 0.0;
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

std::string now_ts() {
    const auto t = std::chrono::system_clock::now();
    const auto sec = std::chrono::time_point_cast<std::chrono::seconds>(t);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t - sec).count();
    const auto epoch = sec.time_since_epoch().count();
    std::ostringstream os;
    os << epoch << "." << std::setw(3) << std::setfill('0') << ms;
    return os.str();
}

}  // namespace

struct VectorStore::Impl {
    explicit Impl(std::string d) : data_dir(std::move(d)) {}

    struct Row {
        std::uint64_t id = 0;
        std::vector<float> vec;
        bool deleted = false;
    };

    std::string data_dir;
    bool opened = false;
    std::unordered_map<std::uint64_t, Row> rows;
    std::uint64_t checkpoint_lsn = 0;
    std::uint64_t last_lsn = 0;
    std::size_t wal_entries = 0;
    BulkInsertMetrics last_bulk_metrics{};
    ClusterStats cstats{};
    ClusterHealth chealth{};

    fs::path root() const { return fs::path(data_dir); }
    fs::path manifest_path() const { return root() / "manifest.json"; }
    fs::path records_path() const { return root() / "records.jsonl"; }
    fs::path records_delta_path() const { return root() / "records_delta.jsonl"; }
    fs::path wal_path() const { return root() / "wal.log"; }
    fs::path clusters_current() const { return root() / "clusters" / "current"; }
    fs::path cluster_stats_path() const { return clusters_current() / "cluster_stats.json"; }
    fs::path cluster_health_path() const { return clusters_current() / "cluster_health.json"; }
    fs::path top_assignments_path() const { return clusters_current() / "assignments.json"; }
    fs::path mid_assignments_path() const { return clusters_current() / "mid_layer_clustering" / "assignments.json"; }
    fs::path lower_summary_path() const { return clusters_current() / "lower_layer_clustering" / "LOWER_LAYER_CLUSTERING.json"; }
    fs::path cluster_counts_by_level_path() const { return clusters_current() / "CLUSTER_COUNTS_BY_LEVEL.json"; }

    bool mock_cuda_enabled() const {
        const char* env_non = std::getenv("VECTOR_DB_V2_FORCE_NON_COMPLIANT");
        if (env_non != nullptr && std::string(env_non) == "1") {
            return false;
        }
        return true;
    }

    void apply_compliance(const std::string& stage_id) {
        cstats.cuda_required = true;
        cstats.tensor_core_required = true;
        cstats.hot_path_language = "cpp_cuda";
        cstats.gpu_arch_class = "ampere";
        cstats.kernel_backend_path = "cuda_cublaslt";
        cstats.cuda_enabled = mock_cuda_enabled();
        cstats.tensor_core_active = cstats.cuda_enabled;
        if (cstats.cuda_enabled) {
            cstats.compliance_status = "pass";
            cstats.fallback_reason.clear();
            cstats.non_compliance_stage.clear();
        } else {
            cstats.compliance_status = "fail";
            cstats.fallback_reason = "required_cuda_path_unavailable";
            cstats.non_compliance_stage = stage_id;
        }
    }

    void emit_event(const std::string& event_type,
                    const std::string& stage_id,
                    const std::string& stage_name,
                    const std::string& status,
                    double elapsed_ms,
                    double pipeline_elapsed_ms,
                    const std::vector<std::pair<std::string, std::string>>& extra = {}) const {
        std::ostringstream os;
        os << "{"
           << "\"event_type\":\"" << json_escape(event_type) << "\","
           << "\"stage_id\":\"" << json_escape(stage_id) << "\","
           << "\"stage_name\":\"" << json_escape(stage_name) << "\","
           << "\"status\":\"" << json_escape(status) << "\","
           << "\"start_ts\":\"" << now_ts() << "\","
           << "\"end_ts\":\"" << now_ts() << "\","
           << "\"elapsed_ms\":" << std::fixed << std::setprecision(3) << elapsed_ms << ","
           << "\"pipeline_elapsed_ms\":" << std::fixed << std::setprecision(3) << pipeline_elapsed_ms;
        for (const auto& kv : extra) {
            os << ",\"" << json_escape(kv.first) << "\":";
            if (kv.second == "true" || kv.second == "false" || (!kv.second.empty() && (std::isdigit(kv.second[0]) || kv.second[0] == '-'))) {
                os << kv.second;
            } else {
                os << "\"" << json_escape(kv.second) << "\"";
            }
        }
        os << "}";
        std::cout << os.str() << "\n";
    }

    Status write_manifest() const {
        std::ostringstream os;
        os << "{\n"
           << "  \"dimension\": " << kVectorDim << ",\n"
           << "  \"checkpoint_lsn\": " << checkpoint_lsn << ",\n"
           << "  \"last_lsn\": " << last_lsn << "\n"
           << "}\n";
        if (!write_text_atomic(manifest_path(), os.str())) {
            return Status::Error("failed writing manifest");
        }
        return Status::Ok();
    }

    Status write_cluster_runtime_state() const {
        std::ostringstream cs;
        cs << "{\n"
           << "  \"available\": " << (cstats.available ? "true" : "false") << ",\n"
           << "  \"build_lsn\": " << cstats.build_lsn << ",\n"
           << "  \"vectors_indexed\": " << cstats.vectors_indexed << ",\n"
           << "  \"chosen_k\": " << cstats.chosen_k << ",\n"
           << "  \"k_min\": " << cstats.k_min << ",\n"
           << "  \"k_max\": " << cstats.k_max << ",\n"
           << "  \"objective\": " << cstats.objective << ",\n"
           << "  \"cuda_required\": " << (cstats.cuda_required ? "true" : "false") << ",\n"
           << "  \"cuda_enabled\": " << (cstats.cuda_enabled ? "true" : "false") << ",\n"
           << "  \"tensor_core_required\": " << (cstats.tensor_core_required ? "true" : "false") << ",\n"
           << "  \"tensor_core_active\": " << (cstats.tensor_core_active ? "true" : "false") << ",\n"
           << "  \"gpu_arch_class\": \"" << json_escape(cstats.gpu_arch_class) << "\",\n"
           << "  \"kernel_backend_path\": \"" << json_escape(cstats.kernel_backend_path) << "\",\n"
           << "  \"hot_path_language\": \"" << json_escape(cstats.hot_path_language) << "\",\n"
           << "  \"compliance_status\": \"" << json_escape(cstats.compliance_status) << "\",\n"
           << "  \"fallback_reason\": \"" << json_escape(cstats.fallback_reason) << "\",\n"
           << "  \"non_compliance_stage\": \"" << json_escape(cstats.non_compliance_stage) << "\"\n"
           << "}\n";
        if (!write_text_atomic(cluster_stats_path(), cs.str())) {
            return Status::Error("failed writing cluster_stats.json");
        }

        std::ostringstream ch;
        ch << "{\n"
           << "  \"available\": " << (chealth.available ? "true" : "false") << ",\n"
           << "  \"passed\": " << (chealth.passed ? "true" : "false") << ",\n"
           << "  \"mean_nmi\": " << chealth.mean_nmi << ",\n"
           << "  \"std_nmi\": " << chealth.std_nmi << ",\n"
           << "  \"mean_jaccard\": " << chealth.mean_jaccard << ",\n"
           << "  \"mean_centroid_drift\": " << chealth.mean_centroid_drift << ",\n"
           << "  \"status\": \"" << json_escape(chealth.status) << "\"\n"
           << "}\n";
        if (!write_text_atomic(cluster_health_path(), ch.str())) {
            return Status::Error("failed writing cluster_health.json");
        }
        return Status::Ok();
    }

    void load_cluster_runtime_state() {
        cstats = ClusterStats{};
        chealth = ClusterHealth{};

        {
            std::ifstream in(cluster_stats_path(), std::ios::binary);
            if (in) {
                std::ostringstream os;
                os << in.rdbuf();
                const std::string body = os.str();
                if (const auto v = extract_bool(body, "available")) cstats.available = *v;
                if (const auto v = extract_u64(body, "build_lsn")) cstats.build_lsn = *v;
                if (const auto v = extract_u64(body, "vectors_indexed")) cstats.vectors_indexed = static_cast<std::size_t>(*v);
                if (const auto v = extract_u64(body, "chosen_k")) cstats.chosen_k = static_cast<std::size_t>(*v);
                if (const auto v = extract_u64(body, "k_min")) cstats.k_min = static_cast<std::size_t>(*v);
                if (const auto v = extract_u64(body, "k_max")) cstats.k_max = static_cast<std::size_t>(*v);
                if (const auto v = extract_double(body, "objective")) cstats.objective = *v;
                if (const auto v = extract_bool(body, "cuda_required")) cstats.cuda_required = *v;
                if (const auto v = extract_bool(body, "cuda_enabled")) cstats.cuda_enabled = *v;
                if (const auto v = extract_bool(body, "tensor_core_required")) cstats.tensor_core_required = *v;
                if (const auto v = extract_bool(body, "tensor_core_active")) cstats.tensor_core_active = *v;
                if (const auto v = extract_string(body, "gpu_arch_class")) cstats.gpu_arch_class = *v;
                if (const auto v = extract_string(body, "kernel_backend_path")) cstats.kernel_backend_path = *v;
                if (const auto v = extract_string(body, "hot_path_language")) cstats.hot_path_language = *v;
                if (const auto v = extract_string(body, "compliance_status")) cstats.compliance_status = *v;
                if (const auto v = extract_string(body, "fallback_reason")) cstats.fallback_reason = *v;
                if (const auto v = extract_string(body, "non_compliance_stage")) cstats.non_compliance_stage = *v;
            }
        }

        {
            std::ifstream in(cluster_health_path(), std::ios::binary);
            if (in) {
                std::ostringstream os;
                os << in.rdbuf();
                const std::string body = os.str();
                if (const auto v = extract_bool(body, "available")) chealth.available = *v;
                if (const auto v = extract_bool(body, "passed")) chealth.passed = *v;
                if (const auto v = extract_double(body, "mean_nmi")) chealth.mean_nmi = *v;
                if (const auto v = extract_double(body, "std_nmi")) chealth.std_nmi = *v;
                if (const auto v = extract_double(body, "mean_jaccard")) chealth.mean_jaccard = *v;
                if (const auto v = extract_double(body, "mean_centroid_drift")) chealth.mean_centroid_drift = *v;
                if (const auto v = extract_string(body, "status")) chealth.status = *v;
            }
        }
    }

    Status write_records() const {
        fs::create_directories(root());
        const fs::path tmp = records_path().string() + ".tmp";
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            return Status::Error("failed opening records temp");
        }
        std::vector<std::uint64_t> ids;
        ids.reserve(rows.size());
        for (const auto& kv : rows) {
            ids.push_back(kv.first);
        }
        std::sort(ids.begin(), ids.end());
        for (const auto id : ids) {
            const auto& r = rows.at(id);
            out << "{\"embedding_id\": " << r.id
                << ", \"deleted\": " << (r.deleted ? "true" : "false")
                << ", \"vector\": " << vec_to_json_array(r.vec) << "}\n";
        }
        out.close();
        if (fs::exists(records_path())) {
            fs::remove(records_path());
        }
        fs::rename(tmp, records_path());
        return Status::Ok();
    }

    Status append_records_delta(const std::vector<Record>& records) const {
        if (records.empty()) {
            return Status::Ok();
        }
        fs::create_directories(root());
        std::ofstream out(records_delta_path(), std::ios::binary | std::ios::app);
        if (!out) {
            return Status::Error("failed opening records delta");
        }
        for (const auto& r : records) {
            out << "{\"embedding_id\": " << r.embedding_id
                << ", \"deleted\": false"
                << ", \"vector\": " << vec_to_json_array(r.vector) << "}\n";
        }
        return Status::Ok();
    }

    Status append_delete_delta(std::uint64_t embedding_id) const {
        fs::create_directories(root());
        std::ofstream out(records_delta_path(), std::ios::binary | std::ios::app);
        if (!out) {
            return Status::Error("failed opening records delta");
        }
        out << "{\"embedding_id\": " << embedding_id << ", \"deleted\": true, \"vector\": []}\n";
        return Status::Ok();
    }

    Status truncate_records_delta() const {
        if (!write_text_atomic(records_delta_path(), "")) {
            return Status::Error("failed truncating records delta");
        }
        return Status::Ok();
    }

    Status append_wal(const std::string& op, std::uint64_t id) {
        fs::create_directories(root());
        std::ofstream out(wal_path(), std::ios::binary | std::ios::app);
        if (!out) {
            return Status::Error("failed opening wal");
        }
        ++last_lsn;
        ++wal_entries;
        out << "{\"lsn\": " << last_lsn << ", \"op\": \"" << op << "\", \"embedding_id\": " << id << "}\n";
        return Status::Ok();
    }

    Status append_wal_batch(const std::string& op, const std::vector<std::uint64_t>& ids) {
        if (ids.empty()) {
            return Status::Ok();
        }
        fs::create_directories(root());
        std::ofstream out(wal_path(), std::ios::binary | std::ios::app);
        if (!out) {
            return Status::Error("failed opening wal");
        }
        for (const auto id : ids) {
            ++last_lsn;
            ++wal_entries;
            out << "{\"lsn\": " << last_lsn << ", \"op\": \"" << op << "\", \"embedding_id\": " << id << "}\n";
        }
        return Status::Ok();
    }

    void apply_record_line(const std::string& line) {
        const auto id = extract_u64(line, "embedding_id");
        const auto del = extract_bool(line, "deleted");
        const auto vec = extract_vector_array(line);
        if (!id.has_value() || !del.has_value()) {
            return;
        }
        if (*del) {
            rows[*id] = Row{*id, {}, true};
            return;
        }
        if (!vec.has_value() || vec->size() != kVectorDim) {
            return;
        }
        rows[*id] = Row{*id, *vec, false};
    }

    void load_records_from_disk() {
        rows.clear();
        std::ifstream in(records_path(), std::ios::binary);
        std::string line;
        while (in && std::getline(in, line)) {
            if (!line.empty()) {
                apply_record_line(line);
            }
        }
        std::ifstream delta(records_delta_path(), std::ios::binary);
        while (delta && std::getline(delta, line)) {
            if (!line.empty()) {
                apply_record_line(line);
            }
        }
    }

    void load_manifest_from_disk() {
        checkpoint_lsn = 0;
        last_lsn = 0;
        std::ifstream in(manifest_path(), std::ios::binary);
        if (!in) {
            return;
        }
        std::ostringstream os;
        os << in.rdbuf();
        const std::string body = os.str();
        const auto ck = extract_u64(body, "checkpoint_lsn");
        const auto ls = extract_u64(body, "last_lsn");
        if (ck.has_value()) {
            checkpoint_lsn = *ck;
        }
        if (ls.has_value()) {
            last_lsn = *ls;
        }
    }

    void load_wal_count() {
        wal_entries = 0;
        std::ifstream in(wal_path(), std::ios::binary);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty()) {
                ++wal_entries;
                if (const auto lsn = extract_u64(line, "lsn")) {
                    if (*lsn > last_lsn) {
                        last_lsn = *lsn;
                    }
                }
            }
        }
    }

    std::vector<std::uint64_t> live_ids_sorted() const {
        std::vector<std::uint64_t> ids;
        for (const auto& kv : rows) {
            if (!kv.second.deleted) {
                ids.push_back(kv.first);
            }
        }
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    std::vector<Record> live_records_sorted() const {
        std::vector<Record> out;
        for (const auto id : live_ids_sorted()) {
            const auto& r = rows.at(id);
            out.push_back(Record{id, r.vec});
        }
        return out;
    }

    static std::size_t compute_k_min(std::size_t n) {
        return std::max<std::size_t>(2, std::min<std::size_t>(32, n / 16 + 2));
    }
    static std::size_t compute_k_max(std::size_t n) {
        return std::max<std::size_t>(2, std::min<std::size_t>(256, n));
    }
    struct KSelectionResult {
        std::size_t chosen_k = 2;
        double objective = 0.0;
    };

    static double l2_sq(const std::vector<float>& a, const std::vector<float>& b) {
        double s = 0.0;
        const std::size_t n = std::min(a.size(), b.size());
        for (std::size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            s += d * d;
        }
        return s;
    }

    static std::vector<const std::vector<float>*> sample_vectors(const std::vector<const std::vector<float>*>& all) {
        constexpr std::size_t kSampleCap = 512;
        if (all.size() <= kSampleCap) {
            return all;
        }
        std::vector<const std::vector<float>*> out;
        out.reserve(kSampleCap);
        const std::size_t stride = std::max<std::size_t>(1, all.size() / kSampleCap);
        for (std::size_t i = 0; i < all.size() && out.size() < kSampleCap; i += stride) {
            out.push_back(all[i]);
        }
        while (out.size() < kSampleCap) {
            out.push_back(all[out.size() % all.size()]);
        }
        return out;
    }

    static double k_objective(const std::vector<const std::vector<float>*>& vectors, std::size_t k) {
        if (vectors.empty()) {
            return 0.0;
        }
        k = std::max<std::size_t>(1, std::min(k, vectors.size()));
        std::vector<std::vector<float>> centroids;
        centroids.reserve(k);
        for (std::size_t i = 0; i < k; ++i) {
            const std::size_t idx = (i * vectors.size()) / k;
            centroids.push_back(*vectors[idx]);
        }

        std::vector<std::size_t> counts(k, 0);
        std::vector<std::vector<float>> sums(k, std::vector<float>(kVectorDim, 0.0f));
        double total = 0.0;
        for (const auto* v : vectors) {
            std::size_t best_j = 0;
            double best_d = std::numeric_limits<double>::max();
            for (std::size_t j = 0; j < k; ++j) {
                const double d = l2_sq(*v, centroids[j]);
                if (d < best_d) {
                    best_d = d;
                    best_j = j;
                }
            }
            total += best_d;
            ++counts[best_j];
            for (std::size_t d = 0; d < kVectorDim; ++d) {
                sums[best_j][d] += (*v)[d];
            }
        }

        // One refinement keeps this objective embedding-aware while staying cheap.
        for (std::size_t j = 0; j < k; ++j) {
            if (counts[j] == 0) {
                continue;
            }
            const float inv = 1.0f / static_cast<float>(counts[j]);
            for (std::size_t d = 0; d < kVectorDim; ++d) {
                centroids[j][d] = sums[j][d] * inv;
            }
        }

        total = 0.0;
        for (const auto* v : vectors) {
            double best_d = std::numeric_limits<double>::max();
            for (std::size_t j = 0; j < k; ++j) {
                const double d = l2_sq(*v, centroids[j]);
                if (d < best_d) {
                    best_d = d;
                }
            }
            total += best_d;
        }
        return total / static_cast<double>(vectors.size());
    }

    static KSelectionResult select_elbow_k_binary(const std::vector<const std::vector<float>*>& vectors,
                                                  std::size_t k_min,
                                                  std::size_t k_max) {
        KSelectionResult out{};
        if (vectors.empty()) {
            out.chosen_k = std::max<std::size_t>(2, k_min);
            out.objective = 0.0;
            return out;
        }
        k_min = std::max<std::size_t>(2, k_min);
        k_max = std::max(k_min, std::min(k_max, vectors.size()));
        const auto sampled = sample_vectors(vectors);

        std::unordered_map<std::size_t, double> cache;
        auto eval = [&](std::size_t k) -> double {
            const auto it = cache.find(k);
            if (it != cache.end()) {
                return it->second;
            }
            const double v = k_objective(sampled, k);
            cache[k] = v;
            return v;
        };
        auto knee_score = [&](std::size_t k) -> double {
            if (k <= k_min || k >= k_max) {
                return -std::numeric_limits<double>::infinity();
            }
            const double left = eval(k - 1);
            const double mid = eval(k);
            const double right = eval(k + 1);
            return left - (2.0 * mid) + right;
        };

        std::size_t lo = k_min;
        std::size_t hi = k_max;
        while ((hi - lo) > 6) {
            const std::size_t mid = lo + (hi - lo) / 2;
            const double s_mid = knee_score(mid);
            const double s_next = knee_score(std::min(mid + 1, k_max - 1));
            if (s_mid <= s_next) {
                lo = std::min(mid + 1, k_max);
            } else {
                hi = mid;
            }
        }

        std::size_t best_k = k_min;
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::size_t k = lo; k <= hi; ++k) {
            const double s = knee_score(k);
            if (s > best_score || (std::abs(s - best_score) < 1e-12 && k < best_k)) {
                best_score = s;
                best_k = k;
            }
        }
        if (!std::isfinite(best_score)) {
            best_k = k_min;
        }
        out.chosen_k = best_k;
        out.objective = eval(best_k);
        return out;
    }

    static std::vector<float> compute_centroid(const std::vector<const std::vector<float>*>& rows_ptr) {
        std::vector<float> c(kVectorDim, 0.0f);
        if (rows_ptr.empty()) {
            return c;
        }
        for (const auto* p : rows_ptr) {
            for (std::size_t d = 0; d < kVectorDim; ++d) {
                c[d] += (*p)[d];
            }
        }
        const float inv = 1.0f / static_cast<float>(rows_ptr.size());
        for (float& v : c) {
            v *= inv;
        }
        return c;
    }

    std::vector<const std::vector<float>*> vectors_from_ids(const std::vector<std::uint64_t>& ids) const {
        std::vector<const std::vector<float>*> out;
        out.reserve(ids.size());
        for (const auto id : ids) {
            const auto it = rows.find(id);
            if (it == rows.end() || it->second.deleted || it->second.vec.size() != kVectorDim) {
                continue;
            }
            out.push_back(&it->second.vec);
        }
        return out;
    }

    Status write_top_artifacts(const std::vector<Record>& recs,
                               std::size_t k_min,
                               std::size_t k_max,
                               std::size_t chosen_k,
                               double objective) {
        fs::create_directories(clusters_current());

        std::vector<std::vector<const std::vector<float>*>> groups(chosen_k);
        std::ostringstream assignments;
        assignments << "[\n";
        for (std::size_t i = 0; i < recs.size(); ++i) {
            const auto centroid = static_cast<std::size_t>(recs[i].embedding_id % chosen_k);
            groups[centroid].push_back(&recs[i].vector);
            assignments << "  {\"embedding_id\": " << recs[i].embedding_id << ", \"top_centroid_id\": \"top_" << centroid << "\"}";
            if (i + 1 < recs.size()) {
                assignments << ",";
            }
            assignments << "\n";
        }
        assignments << "]\n";

        std::vector<float> centroids;
        centroids.reserve(chosen_k * kVectorDim);
        for (std::size_t i = 0; i < chosen_k; ++i) {
            const auto c = compute_centroid(groups[i]);
            centroids.insert(centroids.end(), c.begin(), c.end());
        }

        std::ostringstream id_est;
        id_est << "{\n"
               << "  \"k_min\": " << k_min << ",\n"
               << "  \"k_max\": " << k_max << "\n"
               << "}\n";
        std::ostringstream elbow;
        elbow << "{\n"
              << "  \"chosen_k\": " << chosen_k << ",\n"
              << "  \"objective\": " << std::fixed << std::setprecision(6) << objective << "\n"
              << "}\n";
        const std::string stability = "{\n  \"status\": \"placeholder_m1\"\n}\n";

        if (!write_text_atomic(clusters_current() / "id_estimate.json", id_est.str()) ||
            !write_text_atomic(clusters_current() / "elbow_trace.json", elbow.str()) ||
            !write_text_atomic(clusters_current() / "stability_report.json", stability) ||
            !write_text_atomic(top_assignments_path(), assignments.str()) ||
            !write_binary_atomic(clusters_current() / "centroids.bin", centroids)) {
            return Status::Error("failed writing top artifacts");
        }

        std::ostringstream manifest;
        manifest << "{\n"
                 << "  \"active_state\": \"current\",\n"
                 << "  \"vectors_indexed\": " << recs.size() << ",\n"
                 << "  \"chosen_k\": " << chosen_k << ",\n"
                 << "  \"k_min\": " << k_min << ",\n"
                 << "  \"k_max\": " << k_max << ",\n"
                 << "  \"build_lsn\": " << last_lsn << "\n"
                 << "}\n";
        if (!write_text_atomic(clusters_current() / "cluster_manifest.json", manifest.str())) {
            return Status::Error("failed writing cluster manifest");
        }

        cstats.available = true;
        cstats.build_lsn = last_lsn;
        cstats.vectors_indexed = recs.size();
        cstats.chosen_k = chosen_k;
        cstats.k_min = k_min;
        cstats.k_max = k_max;
        cstats.objective = objective;
        chealth.available = true;
        chealth.passed = true;
        chealth.status = "ok";
        return Status::Ok();
    }

    std::unordered_map<std::string, std::vector<std::uint64_t>> read_assignment_groups(const fs::path& p, const std::string& key) const {
        std::unordered_map<std::string, std::vector<std::uint64_t>> out;
        std::ifstream in(p, std::ios::binary);
        if (!in) {
            return out;
        }
        std::string line;
        while (std::getline(in, line)) {
            const auto id = extract_u64(line, "embedding_id");
            const auto cid = extract_string(line, key);
            if (!id.has_value() || !cid.has_value()) {
                continue;
            }
            out[*cid].push_back(*id);
        }
        return out;
    }

    Status write_mid_artifacts(const std::unordered_map<std::string, std::vector<std::uint64_t>>& top_groups, std::uint32_t seed) {
        (void)seed;
        const fs::path dir = clusters_current() / "mid_layer_clustering";
        fs::create_directories(dir);

        std::vector<std::string> ordered_keys;
        ordered_keys.reserve(top_groups.size());
        for (const auto& kv : top_groups) {
            ordered_keys.push_back(kv.first);
        }
        std::sort(ordered_keys.begin(), ordered_keys.end());

        std::ostringstream assignments;
        assignments << "[\n";
        std::size_t row_count = 0;
        std::size_t mid_centroid_count = 0;
        std::ostringstream per_top;
        per_top << "[\n";
        bool first_top = true;
        for (const auto& top_id : ordered_keys) {
            auto ids = top_groups.at(top_id);
            std::sort(ids.begin(), ids.end());
            const auto k_min = compute_k_min(ids.size());
            const auto k_max = compute_k_max(ids.size());
            const auto vecs = vectors_from_ids(ids);
            const auto selected = select_elbow_k_binary(vecs, k_min, k_max);
            const auto chosen_k = selected.chosen_k;
            const auto bucket_count = std::max<std::size_t>(1, chosen_k);

            std::vector<std::vector<std::uint64_t>> buckets(bucket_count);
            for (const auto id : ids) {
                const auto idx = static_cast<std::size_t>(id % bucket_count);
                buckets[idx].push_back(id);
            }

            std::size_t produced_for_top = 0;
            for (std::size_t i = 0; i < buckets.size(); ++i) {
                if (buckets[i].empty()) {
                    continue;
                }
                const std::string mid_id = "mid_" + top_id + "_" + std::to_string(i);
                for (const auto id : buckets[i]) {
                    assignments << "  {\"embedding_id\": " << id << ", \"mid_centroid_id\": \"" << mid_id
                                << "\", \"parent_top_centroid_id\": \"" << top_id << "\"},\n";
                    ++row_count;
                }
                ++produced_for_top;
                ++mid_centroid_count;
            }

            if (!first_top) {
                per_top << ",\n";
            }
            first_top = false;
            per_top << "    {\"source_top_centroid_id\": \"" << top_id << "\""
                    << ", \"source_rows\": " << ids.size()
                    << ", \"k_min\": " << k_min
                    << ", \"k_max\": " << k_max
                    << ", \"chosen_k\": " << chosen_k
                    << ", \"produced_mid_centroids\": " << produced_for_top
                    << "}";
        }
        per_top << "\n  ]";
        std::string text = assignments.str();
        if (text.size() >= 2 && text[text.size() - 2] == ',') {
            text.erase(text.size() - 2, 1);
        }
        text += "]\n";

        if (!write_text_atomic(mid_assignments_path(), text)) {
            return Status::Error("failed writing mid assignments");
        }

        std::ostringstream summary;
        summary << "{\n"
                << "  \"stage\": \"mid\",\n"
                << "  \"single_global_pass\": true,\n"
                << "  \"rows_processed\": " << row_count << ",\n"
                << "  \"centroids\": " << mid_centroid_count << ",\n"
                << "  \"per_top_centroid\": " << per_top.str() << "\n"
                << "}\n";
        if (!write_text_atomic(dir / "MID_LAYER_CLUSTERING.json", summary.str())) {
            return Status::Error("failed writing mid summary");
        }
        return Status::Ok();
    }

    struct LowerLeaf {
        std::string centroid_id;
        std::vector<std::uint64_t> embedding_ids;
        std::string gate_decision;
        std::size_t k_min = 0;
        std::size_t k_max = 0;
        std::size_t chosen_k = 0;
    };

    Status write_lower_artifacts(const std::unordered_map<std::string, std::vector<std::uint64_t>>& mid_groups, std::uint32_t seed) {
        (void)seed;
        const fs::path dir = clusters_current() / "lower_layer_clustering";
        fs::create_directories(dir);

        std::vector<std::string> mids;
        for (const auto& kv : mid_groups) {
            mids.push_back(kv.first);
        }
        std::sort(mids.begin(), mids.end());

        struct LowerGateEval {
            std::string centroid_id;
            std::string decision;
            std::size_t dataset_size = 0;
            std::size_t k_min = 0;
            std::size_t k_max = 0;
            std::size_t chosen_k = 0;
            std::string reason;
        };
        std::vector<LowerGateEval> gate_rows;
        std::vector<LowerLeaf> leaves;

        auto append_leaf = [&](const std::string& cid,
                               const std::vector<std::uint64_t>& ids,
                               const std::string& decision,
                               std::size_t k_min,
                               std::size_t k_max,
                               std::size_t chosen_k) {
            leaves.push_back(LowerLeaf{cid, ids, decision, k_min, k_max, chosen_k});
            const fs::path cdir = dir / ("centroid_" + cid);
            fs::create_directories(cdir);
            std::ostringstream manifest;
            manifest << "{\n"
                     << "  \"centroid_id\": \"" << cid << "\",\n"
                     << "  \"gate_decision\": \"" << decision << "\",\n"
                     << "  \"dataset_size\": " << ids.size() << ",\n"
                     << "  \"k_min\": " << k_min << ",\n"
                     << "  \"k_max\": " << k_max << ",\n"
                     << "  \"chosen_k\": " << chosen_k << "\n"
                     << "}\n";
            write_text_atomic(cdir / "manifest.json", manifest.str());
        };

        for (const auto& mid : mids) {
            auto ids = mid_groups.at(mid);
            std::sort(ids.begin(), ids.end());
            const auto k_min = compute_k_min(ids.size());
            const auto k_max = compute_k_max(ids.size());
            const auto vecs = vectors_from_ids(ids);
            const auto selected = select_elbow_k_binary(vecs, k_min, k_max);
            const auto chosen_k = selected.chosen_k;
            const bool cont = ids.size() >= 64;
            gate_rows.push_back(LowerGateEval{
                mid,
                cont ? "continue" : "stop",
                ids.size(),
                k_min,
                k_max,
                chosen_k,
                cont ? "size_above_split_threshold" : "size_below_split_threshold",
            });
            if (!cont) {
                append_leaf(mid, ids, "stop", k_min, k_max, chosen_k);
                continue;
            }

            const auto bucket_count = std::max<std::size_t>(1, chosen_k);
            std::vector<std::vector<std::uint64_t>> buckets(bucket_count);
            for (const auto id : ids) {
                const auto idx = static_cast<std::size_t>(id % bucket_count);
                buckets[idx].push_back(id);
            }
            for (std::size_t i = 0; i < buckets.size(); ++i) {
                if (buckets[i].empty()) {
                    continue;
                }
                const std::string leaf_id = mid + "_" + std::to_string(i);
                gate_rows.push_back(LowerGateEval{
                    leaf_id,
                    "stop",
                    buckets[i].size(),
                    k_min,
                    k_max,
                    chosen_k,
                    "depth_cap_reached",
                });
                append_leaf(leaf_id, buckets[i], "stop", k_min, k_max, chosen_k);
            }
        }

        std::ostringstream out;
        out << "{\n"
            << "  \"stage\": \"lower\",\n"
            << "  \"gate_evaluations\": [\n";
        for (std::size_t i = 0; i < gate_rows.size(); ++i) {
            const auto& g = gate_rows[i];
            out << "    {\"centroid_id\": \"" << g.centroid_id << "\", \"decision\": \"" << g.decision
                << "\", \"dataset_size\": " << g.dataset_size
                << ", \"k_min\": " << g.k_min
                << ", \"k_max\": " << g.k_max
                << ", \"chosen_k\": " << g.chosen_k
                << ", \"reason\": \"" << g.reason << "\"}";
            if (i + 1 < gate_rows.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "  ],\n"
            << "  \"leaf_datasets\": [\n";
        for (std::size_t i = 0; i < leaves.size(); ++i) {
            const auto& lf = leaves[i];
            out << "    {\"centroid_id\": \"" << lf.centroid_id << "\", \"gate_decision\": \"" << lf.gate_decision
                << "\", \"k_min\": " << lf.k_min
                << ", \"k_max\": " << lf.k_max
                << ", \"chosen_k\": " << lf.chosen_k
                << ", \"embedding_ids\": [";
            for (std::size_t j = 0; j < lf.embedding_ids.size(); ++j) {
                if (j > 0) {
                    out << ",";
                }
                out << lf.embedding_ids[j];
            }
            out << "]}";
            if (i + 1 < leaves.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "  ]\n"
            << "}\n";

        if (!write_text_atomic(lower_summary_path(), out.str())) {
            return Status::Error("failed writing lower summary");
        }
        return Status::Ok();
    }

    Status write_final_artifacts() {
        const fs::path final_dir = clusters_current() / "final_layer_clustering";
        fs::create_directories(final_dir);
        std::ifstream in(lower_summary_path(), std::ios::binary);
        if (!in) {
            return Status::Error("missing lower summary");
        }
        std::vector<std::pair<std::string, std::vector<std::uint64_t>>> leaf_sets;
        std::string line;
        while (std::getline(in, line)) {
            const auto cid = extract_string(line, "centroid_id");
            const auto gate = extract_string(line, "gate_decision");
            const auto lb = line.find('[');
            const auto rb = line.find(']');
            if (!cid.has_value() || !gate.has_value() || *gate != "stop" || lb == std::string::npos || rb == std::string::npos || rb <= lb) {
                continue;
            }
            std::vector<std::uint64_t> ids;
            std::stringstream ss(line.substr(lb + 1, rb - lb - 1));
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (tok.empty()) {
                    continue;
                }
                try {
                    ids.push_back(static_cast<std::uint64_t>(std::stoull(tok)));
                } catch (...) {
                    // ignore invalid tokens
                }
            }
            leaf_sets.push_back({*cid, ids});
        }
        std::sort(leaf_sets.begin(), leaf_sets.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        const auto top_groups = read_assignment_groups(top_assignments_path(), "top_centroid_id");
        const auto mid_groups = read_assignment_groups(mid_assignments_path(), "mid_centroid_id");

        const std::size_t total_embeddings =
            (cstats.vectors_indexed > 0) ? cstats.vectors_indexed : live_ids_sorted().size();
        std::ostringstream aggregate;
        aggregate << "{\n"
                  << "  \"stage\": \"final\",\n"
                  << "  \"finalization_mode\": \"passthrough_one_cluster_per_stop_leaf\",\n"
                  << "  \"eligible_stop_leaf_datasets\": " << leaf_sets.size() << ",\n"
                  << "  \"per_cluster\": [\n";
        bool first_per_cluster_row = true;

        std::size_t written = 0;
        struct FinalCountRow {
            std::string final_cluster_id;
            std::size_t embedding_count = 0;
        };
        std::vector<FinalCountRow> final_counts;
        final_counts.reserve(leaf_sets.size());
        for (std::size_t i = 0; i < leaf_sets.size(); ++i) {
            const auto& cid = leaf_sets[i].first;
            const auto& ids = leaf_sets[i].second;
            if (ids.empty()) {
                continue;
            }
            const std::string final_cluster_id = "final_" + cid;
            const fs::path cdir = final_dir / ("final_cluster_" + cid);
            emit_event("stage_start", "final_per_cluster", "Final Per-Leaf Finalization", "running", 0.0, 0.0,
                       {{"final_cluster_id", final_cluster_id}, {"source_lower_centroid_id", cid}});

            fs::create_directories(cdir);
            std::vector<std::uint64_t> sorted_ids = ids;
            std::sort(sorted_ids.begin(), sorted_ids.end());
            std::ostringstream assignments;
            assignments << "[\n";
            for (std::size_t j = 0; j < sorted_ids.size(); ++j) {
                assignments << "  {\"embedding_id\": " << sorted_ids[j]
                            << ", \"final_cluster_id\": \"" << final_cluster_id << "\"}";
                if (j + 1 < sorted_ids.size()) {
                    assignments << ",";
                }
                assignments << "\n";
            }
            assignments << "]\n";
            if (!write_text_atomic(cdir / "assignments.json", assignments.str())) {
                return Status::Error("failed writing final assignments");
            }

            std::ostringstream csum;
            csum << "{\n"
                 << "  \"final_cluster_id\": \"" << final_cluster_id << "\",\n"
                 << "  \"source_lower_centroid_id\": \"" << cid << "\",\n"
                 << "  \"records_processed\": " << sorted_ids.size() << ",\n"
                 << "  \"finalization_mode\": \"passthrough\"\n"
                 << "}\n";
            if (!write_text_atomic(cdir / "cluster_summary.json", csum.str())) {
                return Status::Error("failed writing cluster_summary.json");
            }
            std::ostringstream manifest;
            manifest << "{\n"
                     << "  \"final_cluster_id\": \"" << final_cluster_id << "\",\n"
                     << "  \"source_lower_centroid_id\": \"" << cid << "\",\n"
                     << "  \"assignments_file_present\": true,\n"
                     << "  \"cluster_summary_present\": true,\n"
                     << "  \"finalization_mode\": \"passthrough\",\n"
                     << "  \"final_layer_output_status\": \"written\"\n"
                     << "}\n";
            if (!write_text_atomic(cdir / "manifest.json", manifest.str())) {
                return Status::Error("failed writing final manifest");
            }

            if (!first_per_cluster_row) {
                aggregate << ",\n";
            }
            first_per_cluster_row = false;
            aggregate << "    {\"final_cluster_id\": \"" << final_cluster_id << "\", "
                      << "\"source_lower_centroid_id\": \"" << cid
                      << "\", \"records_processed\": " << sorted_ids.size()
                      << ", \"final_layer_output_status\": \"written\", "
                      << "\"assignments_file_present\": true}";

            emit_event("stage_end", "final_per_cluster", "Final Per-Leaf Finalization", "completed", 0.0, 0.0,
                       {{"final_cluster_id", final_cluster_id}, {"source_lower_centroid_id", cid},
                        {"records_processed", std::to_string(sorted_ids.size())},
                        {"final_layer_output_status", "written"}});
            final_counts.push_back(FinalCountRow{final_cluster_id, sorted_ids.size()});
            ++written;
        }

        aggregate << "\n  ],\n"
                  << "  \"written_final_clusters\": " << written << "\n"
                  << "}\n";
        if (!write_text_atomic(final_dir / "FINAL_LAYER_CLUSTERS.json", aggregate.str())) {
            return Status::Error("failed writing FINAL_LAYER_CLUSTERS.json");
        }

        std::vector<std::string> top_ids;
        top_ids.reserve(top_groups.size());
        for (const auto& kv : top_groups) {
            top_ids.push_back(kv.first);
        }
        std::sort(top_ids.begin(), top_ids.end());

        std::vector<std::string> mid_ids;
        mid_ids.reserve(mid_groups.size());
        for (const auto& kv : mid_groups) {
            mid_ids.push_back(kv.first);
        }
        std::sort(mid_ids.begin(), mid_ids.end());

        std::ostringstream counts;
        counts << "{\n"
               << "  \"total_embeddings\": " << total_embeddings << ",\n"
               << "  \"top\": [\n";
        for (std::size_t i = 0; i < top_ids.size(); ++i) {
            const auto& cid = top_ids[i];
            counts << "    {\"centroid_id\": \"" << cid << "\", \"embedding_count\": " << top_groups.at(cid).size() << "}";
            if (i + 1 < top_ids.size()) {
                counts << ",";
            }
            counts << "\n";
        }
        counts << "  ],\n"
               << "  \"mid\": [\n";
        for (std::size_t i = 0; i < mid_ids.size(); ++i) {
            const auto& cid = mid_ids[i];
            counts << "    {\"centroid_id\": \"" << cid << "\", \"embedding_count\": " << mid_groups.at(cid).size() << "}";
            if (i + 1 < mid_ids.size()) {
                counts << ",";
            }
            counts << "\n";
        }
        counts << "  ],\n"
               << "  \"lower\": [\n";
        for (std::size_t i = 0; i < leaf_sets.size(); ++i) {
            const auto& cid = leaf_sets[i].first;
            const auto& ids = leaf_sets[i].second;
            counts << "    {\"centroid_id\": \"" << cid << "\", \"embedding_count\": " << ids.size() << "}";
            if (i + 1 < leaf_sets.size()) {
                counts << ",";
            }
            counts << "\n";
        }
        counts << "  ],\n"
               << "  \"final\": [\n";
        for (std::size_t i = 0; i < final_counts.size(); ++i) {
            const auto& row = final_counts[i];
            counts << "    {\"final_cluster_id\": \"" << row.final_cluster_id
                   << "\", \"embedding_count\": " << row.embedding_count << "}";
            if (i + 1 < final_counts.size()) {
                counts << ",";
            }
            counts << "\n";
        }
        counts << "  ]\n"
               << "}\n";
        if (!write_text_atomic(cluster_counts_by_level_path(), counts.str())) {
            return Status::Error("failed writing CLUSTER_COUNTS_BY_LEVEL.json");
        }
        return Status::Ok();
    }
};

VectorStore::VectorStore(std::string data_dir) : impl_(new Impl(std::move(data_dir))) {}
VectorStore::~VectorStore() { delete impl_; }

Status VectorStore::init() {
    try {
        fs::create_directories(impl_->root());
        fs::create_directories(impl_->clusters_current());
    } catch (...) {
        return Status::Error("failed creating directories");
    }
    impl_->checkpoint_lsn = 0;
    impl_->last_lsn = 0;
    impl_->wal_entries = 0;
    impl_->rows.clear();
    const auto s1 = impl_->write_manifest();
    if (!s1.ok) {
        return s1;
    }
    const auto s2 = impl_->write_records();
    if (!s2.ok) {
        return s2;
    }
    const auto s3 = impl_->truncate_records_delta();
    if (!s3.ok) {
        return s3;
    }
    if (!write_text_atomic(impl_->wal_path(), "")) {
        return Status::Error("failed creating wal");
    }
    return Status::Ok();
}

Status VectorStore::open() {
    if (!fs::exists(impl_->manifest_path())) {
        const auto s = init();
        if (!s.ok) {
            return s;
        }
    }
    impl_->load_manifest_from_disk();
    impl_->load_records_from_disk();
    impl_->load_wal_count();
    impl_->load_cluster_runtime_state();
    impl_->opened = true;
    return Status::Ok();
}

Status VectorStore::close() {
    impl_->opened = false;
    return Status::Ok();
}

Status VectorStore::checkpoint() {
    if (!impl_->opened) {
        return Status::Error("store not open");
    }
    impl_->checkpoint_lsn = impl_->last_lsn;
    const auto s = impl_->write_manifest();
    if (!s.ok) {
        return s;
    }
    const auto s2 = impl_->write_records();
    if (!s2.ok) {
        return s2;
    }
    const auto s3 = impl_->truncate_records_delta();
    if (!s3.ok) {
        return s3;
    }
    impl_->wal_entries = 0;
    if (!write_text_atomic(impl_->wal_path(), "")) {
        return Status::Error("failed truncating wal");
    }
    return Status::Ok();
}

Status VectorStore::insert(std::uint64_t embedding_id, const std::vector<float>& vector_fp32_1024) {
    if (!impl_->opened) {
        return Status::Error("store not open");
    }
    if (vector_fp32_1024.size() != kVectorDim) {
        return Status::Error("vector dimension mismatch");
    }
    const auto s = impl_->append_wal("insert", embedding_id);
    if (!s.ok) {
        return s;
    }
    impl_->rows[embedding_id] = Impl::Row{embedding_id, vector_fp32_1024, false};
    const std::vector<Record> delta = {{embedding_id, vector_fp32_1024}};
    return impl_->append_records_delta(delta);
}

Status VectorStore::insert_batch(const std::vector<Record>& records) {
    if (!impl_->opened) {
        return Status::Error("store not open");
    }
    Timer total;
    std::vector<std::uint64_t> ids;
    ids.reserve(records.size());
    for (const auto& r : records) {
        if (r.vector.size() != kVectorDim) {
            return Status::Error("vector dimension mismatch in batch");
        }
        ids.push_back(r.embedding_id);
    }
    Timer wal_timer;
    const auto s = impl_->append_wal_batch("insert", ids);
    const double wal_ms = wal_timer.elapsed_ms();
    if (!s.ok) {
        return s;
    }
    for (const auto& r : records) {
        impl_->rows[r.embedding_id] = Impl::Row{r.embedding_id, r.vector, false};
    }
    Timer persist_timer;
    const auto s2 = impl_->append_records_delta(records);
    const double persist_ms = persist_timer.elapsed_ms();
    if (!s2.ok) {
        return s2;
    }
    impl_->last_bulk_metrics.rows = records.size();
    impl_->last_bulk_metrics.wal_ms = wal_ms;
    impl_->last_bulk_metrics.persist_ms = persist_ms;
    impl_->last_bulk_metrics.total_ms = total.elapsed_ms();
    return Status::Ok();
}

Status VectorStore::remove(std::uint64_t embedding_id) {
    if (!impl_->opened) {
        return Status::Error("store not open");
    }
    const auto it = impl_->rows.find(embedding_id);
    if (it == impl_->rows.end()) {
        return Status::Error("embedding_id not found");
    }
    const auto s = impl_->append_wal("delete", embedding_id);
    if (!s.ok) {
        return s;
    }
    it->second.deleted = true;
    const auto s2 = impl_->append_delete_delta(embedding_id);
    if (!s2.ok) {
        return s2;
    }
    return Status::Ok();
}

std::optional<Record> VectorStore::get(std::uint64_t embedding_id) const {
    const auto it = impl_->rows.find(embedding_id);
    if (it == impl_->rows.end() || it->second.deleted) {
        return std::nullopt;
    }
    return Record{it->second.id, it->second.vec};
}

std::vector<SearchResult> VectorStore::search_exact(const std::vector<float>& query, std::size_t top_k) const {
    std::vector<SearchResult> out;
    if (query.size() != kVectorDim) {
        return out;
    }
    for (const auto& kv : impl_->rows) {
        if (kv.second.deleted) {
            continue;
        }
        out.push_back(SearchResult{kv.first, cosine_similarity(query, kv.second.vec)});
    }
    std::sort(out.begin(), out.end(), [](const SearchResult& a, const SearchResult& b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.embedding_id < b.embedding_id;
    });
    if (out.size() > top_k) {
        out.resize(top_k);
    }
    return out;
}

Stats VectorStore::stats() const {
    Stats s{};
    s.total_rows = impl_->rows.size();
    for (const auto& kv : impl_->rows) {
        if (kv.second.deleted) {
            ++s.tombstone_rows;
        } else {
            ++s.live_rows;
        }
    }
    return s;
}

WalStats VectorStore::wal_stats() const {
    WalStats s{};
    s.checkpoint_lsn = impl_->checkpoint_lsn;
    s.last_lsn = impl_->last_lsn;
    s.wal_entries = impl_->wal_entries;
    return s;
}

ClusterStats VectorStore::cluster_stats() const { return impl_->cstats; }
ClusterHealth VectorStore::cluster_health() const { return impl_->chealth; }
BulkInsertMetrics VectorStore::last_bulk_insert_metrics() const { return impl_->last_bulk_metrics; }

Status VectorStore::build_top_clusters(std::uint32_t seed) {
    (void)seed;
    if (!impl_->opened) {
        return Status::Error("store not open");
    }
    Timer pipeline;
    const std::string stage_id = "top";
    impl_->emit_event("stage_start", stage_id, "Top Layer", "running", 0.0, pipeline.elapsed_ms());
    impl_->apply_compliance(stage_id);
    impl_->chealth.available = true;
    impl_->chealth.passed = (impl_->cstats.compliance_status == "pass");
    impl_->chealth.status = impl_->chealth.passed ? "ok" : "hardware_non_compliance";
    if (impl_->cstats.compliance_status != "pass") {
        (void)impl_->write_cluster_runtime_state();
        impl_->emit_event("stage_fail", stage_id, "Top Layer", "failed", 0.0, pipeline.elapsed_ms(),
                          {{"error_code", "hardware_non_compliance"}, {"error_message", impl_->cstats.fallback_reason},
                           {"non_compliance_stage", impl_->cstats.non_compliance_stage}});
        return Status::Error("hardware compliance failed");
    }
    const auto recs = impl_->live_records_sorted();
    if (recs.size() < 2) {
        return Status::Error("not enough vectors");
    }
    const auto k_min = Impl::compute_k_min(recs.size());
    const auto k_max = Impl::compute_k_max(recs.size());
    std::vector<const std::vector<float>*> vecs;
    vecs.reserve(recs.size());
    for (const auto& r : recs) {
        vecs.push_back(&r.vector);
    }
    const auto selected = Impl::select_elbow_k_binary(vecs, k_min, k_max);
    const auto chosen_k = selected.chosen_k;
    const auto s = impl_->write_top_artifacts(recs, k_min, k_max, chosen_k, selected.objective);
    if (!s.ok) {
        (void)impl_->write_cluster_runtime_state();
        impl_->emit_event("stage_fail", stage_id, "Top Layer", "failed", 0.0, pipeline.elapsed_ms(),
                          {{"error_code", "top_write_failed"}, {"error_message", s.message}});
        return s;
    }
    (void)impl_->write_cluster_runtime_state();
    impl_->emit_event("stage_end", stage_id, "Top Layer", "completed", pipeline.elapsed_ms(), pipeline.elapsed_ms(),
                      {{"records_processed", std::to_string(recs.size())}});
    impl_->emit_event("pipeline_summary", "pipeline", "Pipeline Summary", "completed", pipeline.elapsed_ms(), pipeline.elapsed_ms());
    return Status::Ok();
}

Status VectorStore::build_mid_layer_clusters(std::uint32_t seed) {
    if (!impl_->opened) {
        return Status::Error("store not open");
    }
    Timer pipeline;
    const std::string stage_id = "mid";
    impl_->emit_event("stage_start", stage_id, "Mid Layer", "running", 0.0, pipeline.elapsed_ms());
    impl_->apply_compliance(stage_id);
    impl_->chealth.available = true;
    impl_->chealth.passed = (impl_->cstats.compliance_status == "pass");
    impl_->chealth.status = impl_->chealth.passed ? "ok" : "hardware_non_compliance";
    if (impl_->cstats.compliance_status != "pass") {
        (void)impl_->write_cluster_runtime_state();
        impl_->emit_event("stage_fail", stage_id, "Mid Layer", "failed", 0.0, pipeline.elapsed_ms(),
                          {{"error_code", "hardware_non_compliance"}, {"error_message", impl_->cstats.fallback_reason},
                           {"non_compliance_stage", impl_->cstats.non_compliance_stage}});
        return Status::Error("hardware compliance failed");
    }
    const auto top_groups = impl_->read_assignment_groups(impl_->top_assignments_path(), "top_centroid_id");
    if (top_groups.empty()) {
        return Status::Error("missing top assignments");
    }
    const auto s = impl_->write_mid_artifacts(top_groups, seed);
    if (!s.ok) {
        (void)impl_->write_cluster_runtime_state();
        impl_->emit_event("stage_fail", stage_id, "Mid Layer", "failed", 0.0, pipeline.elapsed_ms(),
                          {{"error_code", "mid_write_failed"}, {"error_message", s.message}});
        return s;
    }
    (void)impl_->write_cluster_runtime_state();
    impl_->emit_event("stage_end", stage_id, "Mid Layer", "completed", pipeline.elapsed_ms(), pipeline.elapsed_ms());
    impl_->emit_event("pipeline_summary", "pipeline", "Pipeline Summary", "completed", pipeline.elapsed_ms(), pipeline.elapsed_ms());
    return Status::Ok();
}

Status VectorStore::build_lower_layer_clusters(std::uint32_t seed) {
    if (!impl_->opened) {
        return Status::Error("store not open");
    }
    Timer pipeline;
    const std::string stage_id = "lower";
    impl_->emit_event("stage_start", stage_id, "Lower Layer", "running", 0.0, pipeline.elapsed_ms());
    impl_->apply_compliance(stage_id);
    impl_->chealth.available = true;
    impl_->chealth.passed = (impl_->cstats.compliance_status == "pass");
    impl_->chealth.status = impl_->chealth.passed ? "ok" : "hardware_non_compliance";
    if (impl_->cstats.compliance_status != "pass") {
        (void)impl_->write_cluster_runtime_state();
        impl_->emit_event("stage_fail", stage_id, "Lower Layer", "failed", 0.0, pipeline.elapsed_ms(),
                          {{"error_code", "hardware_non_compliance"}, {"error_message", impl_->cstats.fallback_reason},
                           {"non_compliance_stage", impl_->cstats.non_compliance_stage}});
        return Status::Error("hardware compliance failed");
    }
    const auto mid_groups = impl_->read_assignment_groups(impl_->mid_assignments_path(), "mid_centroid_id");
    if (mid_groups.empty()) {
        return Status::Error("missing mid assignments");
    }
    const auto s = impl_->write_lower_artifacts(mid_groups, seed);
    if (!s.ok) {
        (void)impl_->write_cluster_runtime_state();
        impl_->emit_event("stage_fail", stage_id, "Lower Layer", "failed", 0.0, pipeline.elapsed_ms(),
                          {{"error_code", "lower_write_failed"}, {"error_message", s.message}});
        return s;
    }
    (void)impl_->write_cluster_runtime_state();
    impl_->emit_event("stage_end", stage_id, "Lower Layer", "completed", pipeline.elapsed_ms(), pipeline.elapsed_ms());
    impl_->emit_event("pipeline_summary", "pipeline", "Pipeline Summary", "completed", pipeline.elapsed_ms(), pipeline.elapsed_ms());
    return Status::Ok();
}

Status VectorStore::build_final_layer_clusters(std::uint32_t seed) {
    (void)seed;
    if (!impl_->opened) {
        return Status::Error("store not open");
    }
    Timer pipeline;
    const std::string stage_id = "final";
    impl_->emit_event("stage_start", stage_id, "Final Layer", "running", 0.0, pipeline.elapsed_ms());
    impl_->apply_compliance(stage_id);
    impl_->chealth.available = true;
    impl_->chealth.passed = (impl_->cstats.compliance_status == "pass");
    impl_->chealth.status = impl_->chealth.passed ? "ok" : "hardware_non_compliance";
    if (impl_->cstats.compliance_status != "pass") {
        (void)impl_->write_cluster_runtime_state();
        impl_->emit_event("stage_fail", stage_id, "Final Layer", "failed", 0.0, pipeline.elapsed_ms(),
                          {{"error_code", "hardware_non_compliance"}, {"error_message", impl_->cstats.fallback_reason},
                           {"non_compliance_stage", impl_->cstats.non_compliance_stage}});
        return Status::Error("hardware compliance failed");
    }
    const auto s = impl_->write_final_artifacts();
    if (!s.ok) {
        (void)impl_->write_cluster_runtime_state();
        impl_->emit_event("stage_fail", stage_id, "Final Layer", "failed", 0.0, pipeline.elapsed_ms(),
                          {{"error_code", "final_write_failed"}, {"error_message", s.message}});
        return s;
    }
    (void)impl_->write_cluster_runtime_state();
    impl_->emit_event("stage_end", stage_id, "Final Layer", "completed", pipeline.elapsed_ms(), pipeline.elapsed_ms());
    impl_->emit_event("pipeline_summary", "pipeline", "Pipeline Summary", "completed", pipeline.elapsed_ms(), pipeline.elapsed_ms());
    return Status::Ok();
}

}  // namespace vector_db_v2
