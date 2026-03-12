#include "vector_db/vector_store.hpp"
#include "vector_db/clustering.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace fs = std::filesystem;

namespace vector_db {

namespace {

std::string trim_copy(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])) != 0) {
        ++b;
    }
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) {
        --e;
    }
    return s.substr(b, e - b);
}

bool looks_like_json(const std::string& s) {
    const std::string t = trim_copy(s);
    if (t.empty()) {
        return false;
    }
    const bool obj = t.front() == '{' && t.back() == '}';
    const bool arr = t.front() == '[' && t.back() == ']';
    return obj || arr;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
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

std::string json_unescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '\\' && i + 1 < s.size()) {
            const char n = s[i + 1];
            switch (n) {
                case '\\':
                    out.push_back('\\');
                    ++i;
                    continue;
                case '"':
                    out.push_back('"');
                    ++i;
                    continue;
                case 'n':
                    out.push_back('\n');
                    ++i;
                    continue;
                case 'r':
                    out.push_back('\r');
                    ++i;
                    continue;
                case 't':
                    out.push_back('\t');
                    ++i;
                    continue;
                default:
                    break;
            }
        }
        out.push_back(c);
    }
    return out;
}

std::vector<std::string> split_csv_numbers(const std::string& csv) {
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            out.push_back(trim_copy(token));
        }
    }
    return out;
}

std::optional<std::uint64_t> extract_u64_field(const std::string& text, const std::string& key) {
    const auto key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto n0 = text.find_first_of("0123456789", key_pos);
    if (n0 == std::string::npos) {
        return std::nullopt;
    }
    const auto n1 = text.find_first_not_of("0123456789", n0);
    try {
        return static_cast<std::uint64_t>(std::stoull(text.substr(n0, n1 - n0)));
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::string> extract_string_field(const std::string& text, const std::string& key) {
    const auto key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto colon = text.find(':', key_pos);
    if (colon == std::string::npos) {
        return std::nullopt;
    }
    const auto q1 = text.find('"', colon);
    if (q1 == std::string::npos) {
        return std::nullopt;
    }
    std::size_t q2 = std::string::npos;
    bool escaped = false;
    for (std::size_t i = q1 + 1; i < text.size(); ++i) {
        const char c = text[i];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            q2 = i;
            break;
        }
    }
    if (q2 == std::string::npos || q2 <= q1) {
        return std::nullopt;
    }
    return text.substr(q1 + 1, q2 - q1 - 1);
}

std::optional<double> extract_double_field(const std::string& text, const std::string& key) {
    const auto key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto n0 = text.find_first_of("-0123456789", key_pos);
    if (n0 == std::string::npos) {
        return std::nullopt;
    }
    const auto n1 = text.find_first_not_of("0123456789+-.eE", n0);
    try {
        return std::stod(text.substr(n0, n1 - n0));
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<bool> extract_bool_field(const std::string& text, const std::string& key) {
    const auto key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto t = text.find("true", key_pos);
    if (t != std::string::npos) {
        return true;
    }
    const auto f = text.find("false", key_pos);
    if (f != std::string::npos) {
        return false;
    }
    return std::nullopt;
}

bool env_flag_enabled(const char* name, bool default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }
    std::string v(raw);
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

std::size_t env_size_value(const char* name, std::size_t default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }
    try {
        return static_cast<std::size_t>(std::stoull(raw));
    } catch (...) {
        return default_value;
    }
}

Status write_text_atomic(const fs::path& path, const std::string& content) {
    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            return Status::Error("failed writing temp file: " + tmp.string());
        }
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
        if (!out.good()) {
            return Status::Error("failed writing content to temp file: " + tmp.string());
        }
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(path, ec);
        ec.clear();
        fs::rename(tmp, path, ec);
        if (ec) {
            return Status::Error("failed renaming temp file into place: " + path.string());
        }
    }
    return Status::Ok();
}

}  // namespace

struct VectorStore::Impl {
    explicit Impl(std::string root)
        : data_dir(std::move(root)),
          open_reload_cache_enabled(env_flag_enabled("VECTOR_DB_OPEN_RELOAD_CACHE", false)) {}

    struct Entry {
        std::uint64_t id = 0;
        std::size_t row = 0;
        bool deleted = false;
        std::string metadata_json = "{}";
    };

    struct WalRecord {
        std::uint64_t lsn = 0;
        std::string op;
        std::uint64_t id = 0;
        std::string metadata_json;
        std::vector<float> vector_fp32;
    };

    std::string data_dir;
    bool opened = false;
    bool replay_mode = false;
    std::uint64_t active_segment_id = 1;
    std::uint64_t checkpoint_lsn = 0;
    std::uint64_t last_lsn = 0;
    std::size_t total_rows = 0;

    std::unordered_map<std::uint64_t, Entry> entries;
    std::vector<DirtyRange> dirty_ranges;
    ClusterStats cluster_stats_cache;
    ClusterHealth cluster_health_cache;
    bool open_reload_cache_enabled = false;

    struct FileSig {
        bool exists = false;
        std::uint64_t size = 0;
        std::int64_t mtime_ns = 0;
    };

    struct OpenSignature {
        std::uint64_t active_segment_id = 0;
        FileSig manifest;
        FileSig dirty_ranges;
        FileSig cluster_manifest;
        FileSig wal;
        FileSig ids;
        FileSig meta;
        FileSig tomb;
        bool valid = false;
    };

    struct LiveVectorLoadResult {
        std::vector<std::pair<std::uint64_t, std::vector<float>>> vectors_by_id;
        std::vector<float> packed_row_major;
        std::size_t bytes_read = 0;
        std::size_t contiguous_spans = 0;
        std::size_t sparse_reads = 0;
        bool sparse_fallback_used = false;
        bool async_double_buffer_used = false;
    };

    struct SecondLevelCentroidSummary {
        std::uint32_t centroid_id = 0;
        std::size_t source_vectors = 0;
        std::size_t vectors_indexed = 0;
        bool processed = false;
        std::string skipped_reason;
        fs::path output_dir;
        ClusterStats stats;
        ClusterHealth health;
    };

    OpenSignature last_open_signature;

    fs::path root() const { return fs::path(data_dir); }
    fs::path segments_dir() const { return root() / "segments"; }
    fs::path manifest_path() const { return root() / "manifest.json"; }
    fs::path dirty_ranges_path() const { return root() / "dirty_ranges.json"; }
    fs::path wal_path() const { return root() / "wal.log"; }
    fs::path clusters_root() const { return root() / "clusters" / "initial"; }
    fs::path cluster_manifest_path() const { return clusters_root() / "cluster_manifest.json"; }
    fs::path cluster_version_dir(std::uint64_t version) const {
        return clusters_root() / ("v" + std::to_string(version));
    }
    fs::path second_level_root(std::uint64_t parent_version) const {
        return cluster_version_dir(parent_version) / "second_level_clustering";
    }

    fs::path seg_base(std::uint64_t seg_id) const {
        return segments_dir() / ("seg_" + std::to_string(seg_id));
    }

    fs::path seg_vec(std::uint64_t seg_id) const { return seg_base(seg_id).string() + ".vec"; }
    fs::path seg_ids(std::uint64_t seg_id) const { return seg_base(seg_id).string() + ".ids"; }
    fs::path seg_meta(std::uint64_t seg_id) const { return seg_base(seg_id).string() + ".meta.jsonl"; }
    fs::path seg_tomb(std::uint64_t seg_id) const { return seg_base(seg_id).string() + ".tomb"; }

    static FileSig file_sig(const fs::path& p) {
        FileSig sig{};
        std::error_code ec;
        sig.exists = fs::exists(p, ec) && !ec;
        if (!sig.exists) {
            return sig;
        }
        ec.clear();
        sig.size = static_cast<std::uint64_t>(fs::file_size(p, ec));
        if (ec) {
            sig.size = 0;
            ec.clear();
        }
        const auto ft = fs::last_write_time(p, ec);
        if (!ec) {
            sig.mtime_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(ft.time_since_epoch()).count();
        }
        return sig;
    }

    OpenSignature capture_open_signature() const {
        OpenSignature sig{};
        sig.active_segment_id = active_segment_id;
        sig.manifest = file_sig(manifest_path());
        sig.dirty_ranges = file_sig(dirty_ranges_path());
        sig.cluster_manifest = file_sig(cluster_manifest_path());
        sig.wal = file_sig(wal_path());
        sig.ids = file_sig(seg_ids(active_segment_id));
        sig.meta = file_sig(seg_meta(active_segment_id));
        sig.tomb = file_sig(seg_tomb(active_segment_id));
        sig.valid = true;
        return sig;
    }

    bool open_signature_unchanged() const {
        if (!last_open_signature.valid || !open_reload_cache_enabled) {
            return false;
        }
        const OpenSignature now = capture_open_signature();
        return now.active_segment_id == last_open_signature.active_segment_id
            && now.manifest.exists == last_open_signature.manifest.exists
            && now.manifest.size == last_open_signature.manifest.size
            && now.manifest.mtime_ns == last_open_signature.manifest.mtime_ns
            && now.dirty_ranges.exists == last_open_signature.dirty_ranges.exists
            && now.dirty_ranges.size == last_open_signature.dirty_ranges.size
            && now.dirty_ranges.mtime_ns == last_open_signature.dirty_ranges.mtime_ns
            && now.cluster_manifest.exists == last_open_signature.cluster_manifest.exists
            && now.cluster_manifest.size == last_open_signature.cluster_manifest.size
            && now.cluster_manifest.mtime_ns == last_open_signature.cluster_manifest.mtime_ns
            && now.wal.exists == last_open_signature.wal.exists
            && now.wal.size == last_open_signature.wal.size
            && now.wal.mtime_ns == last_open_signature.wal.mtime_ns
            && now.ids.exists == last_open_signature.ids.exists
            && now.ids.size == last_open_signature.ids.size
            && now.ids.mtime_ns == last_open_signature.ids.mtime_ns
            && now.meta.exists == last_open_signature.meta.exists
            && now.meta.size == last_open_signature.meta.size
            && now.meta.mtime_ns == last_open_signature.meta.mtime_ns
            && now.tomb.exists == last_open_signature.tomb.exists
            && now.tomb.size == last_open_signature.tomb.size
            && now.tomb.mtime_ns == last_open_signature.tomb.mtime_ns;
    }

    Status ensure_dirs() const {
        std::error_code ec;
        fs::create_directories(root(), ec);
        if (ec) {
            return Status::Error("failed creating data root: " + root().string());
        }
        fs::create_directories(segments_dir(), ec);
        if (ec) {
            return Status::Error("failed creating segments dir: " + segments_dir().string());
        }
        fs::create_directories(clusters_root(), ec);
        if (ec) {
            return Status::Error("failed creating clusters dir: " + clusters_root().string());
        }
        return Status::Ok();
    }

    Status write_manifest() const {
        const Stats st = current_stats();
        std::ostringstream os;
        os << "{\n";
        os << "  \"schema_version\": 1,\n";
        os << "  \"dimension\": " << kVectorDim << ",\n";
        os << "  \"active_segment_id\": " << active_segment_id << ",\n";
        os << "  \"checkpoint_lsn\": " << checkpoint_lsn << ",\n";
        os << "  \"segments\": [" << active_segment_id << "],\n";
        os << "  \"total_rows\": " << st.total_rows << ",\n";
        os << "  \"live_rows\": " << st.live_rows << ",\n";
        os << "  \"tombstone_rows\": " << st.tombstone_rows << "\n";
        os << "}\n";
        return write_text_atomic(manifest_path(), os.str());
    }

    Status write_dirty_ranges() const {
        std::ostringstream os;
        os << "{\n  \"dirty_ranges\": [\n";
        for (std::size_t i = 0; i < dirty_ranges.size(); ++i) {
            const auto& r = dirty_ranges[i];
            os << "    {\"segment_id\": " << r.segment_id
               << ", \"start_row\": " << r.start_row
               << ", \"end_row\": " << r.end_row
               << ", \"reason\": \"" << json_escape(r.reason) << "\"}";
            if (i + 1 < dirty_ranges.size()) {
                os << ",";
            }
            os << "\n";
        }
        os << "  ]\n}\n";
        return write_text_atomic(dirty_ranges_path(), os.str());
    }

    Stats current_stats() const {
        Stats st{};
        st.total_rows = total_rows;
        st.segments = 1;
        std::size_t tomb = 0;
        for (const auto& kv : entries) {
            if (kv.second.deleted) {
                ++tomb;
            }
        }
        st.tombstone_rows = tomb;
        st.live_rows = entries.size() - tomb;
        st.dirty_ranges = dirty_ranges.size();
        return st;
    }

    void record_dirty(std::size_t row, const std::string& reason) {
        if (!dirty_ranges.empty()) {
            DirtyRange& last = dirty_ranges.back();
            if (last.segment_id == active_segment_id
                && last.reason == reason
                && row >= last.start_row
                && row <= (last.end_row + 1)) {
                if (row > last.end_row) {
                    last.end_row = row;
                }
                return;
            }
        }
        dirty_ranges.push_back(DirtyRange{
            active_segment_id,
            row,
            row,
            reason,
        });
    }

    std::uint64_t next_lsn() {
        if (last_lsn < checkpoint_lsn) {
            last_lsn = checkpoint_lsn;
        }
        last_lsn += 1;
        return last_lsn;
    }

    std::uint64_t now_ms() const {
        const auto now = std::chrono::system_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
        return static_cast<std::uint64_t>(ms.count());
    }

    std::string encode_vector_csv(const std::vector<float>& vec) const {
        std::ostringstream os;
        for (std::size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) {
                os << ",";
            }
            os << vec[i];
        }
        return os.str();
    }

    std::optional<std::vector<float>> decode_vector_csv(const std::string& csv) const {
        const auto parts = split_csv_numbers(csv);
        if (parts.size() != kVectorDim) {
            return std::nullopt;
        }
        std::vector<float> vec(kVectorDim, 0.0f);
        for (std::size_t i = 0; i < parts.size(); ++i) {
            try {
                vec[i] = std::stof(parts[i]);
            } catch (...) {
                return std::nullopt;
            }
        }
        return vec;
    }

    Status append_wal_insert(std::uint64_t id, const std::vector<float>& vec, const std::string& metadata_json) {
        const std::uint64_t lsn = next_lsn();
        std::ofstream out(wal_path(), std::ios::binary | std::ios::app);
        if (!out) {
            return Status::Error("failed opening wal for insert");
        }
        out << "{\"lsn\":" << lsn
            << ",\"op\":\"INSERT\""
            << ",\"id\":" << id
            << ",\"metadata\":\"" << json_escape(metadata_json) << "\""
            << ",\"vector\":\"" << json_escape(encode_vector_csv(vec)) << "\""
            << ",\"ts_ms\":" << now_ms()
            << "}\n";
        if (!out.good()) {
            return Status::Error("failed appending wal insert");
        }
        return Status::Ok();
    }

    Status append_wal_insert_batch(const std::vector<Record>& records) {
        if (records.empty()) {
            return Status::Ok();
        }
        std::ofstream out(wal_path(), std::ios::binary | std::ios::app);
        if (!out) {
            return Status::Error("failed opening wal for batch insert");
        }
        for (const auto& rec : records) {
            const std::uint64_t lsn = next_lsn();
            out << "{\"lsn\":" << lsn
                << ",\"op\":\"INSERT\""
                << ",\"id\":" << rec.id
                << ",\"metadata\":\"" << json_escape(rec.metadata_json) << "\""
                << ",\"vector\":\"" << json_escape(encode_vector_csv(rec.vector_fp32)) << "\""
                << ",\"ts_ms\":" << now_ms()
                << "}\n";
            if (!out.good()) {
                return Status::Error("failed appending wal batch insert");
            }
        }
        return Status::Ok();
    }

    Status append_wal_delete(std::uint64_t id) {
        const std::uint64_t lsn = next_lsn();
        std::ofstream out(wal_path(), std::ios::binary | std::ios::app);
        if (!out) {
            return Status::Error("failed opening wal for delete");
        }
        out << "{\"lsn\":" << lsn
            << ",\"op\":\"DELETE\""
            << ",\"id\":" << id
            << ",\"ts_ms\":" << now_ms()
            << "}\n";
        if (!out.good()) {
            return Status::Error("failed appending wal delete");
        }
        return Status::Ok();
    }

    Status append_wal_update_meta(std::uint64_t id, const std::string& metadata_json) {
        const std::uint64_t lsn = next_lsn();
        std::ofstream out(wal_path(), std::ios::binary | std::ios::app);
        if (!out) {
            return Status::Error("failed opening wal for metadata update");
        }
        out << "{\"lsn\":" << lsn
            << ",\"op\":\"UPDATE_META\""
            << ",\"id\":" << id
            << ",\"metadata\":\"" << json_escape(metadata_json) << "\""
            << ",\"ts_ms\":" << now_ms()
            << "}\n";
        if (!out.good()) {
            return Status::Error("failed appending wal metadata update");
        }
        return Status::Ok();
    }

    std::optional<WalRecord> parse_wal_line(const std::string& line) const {
        if (trim_copy(line).empty()) {
            return std::nullopt;
        }
        const auto lsn = extract_u64_field(line, "lsn");
        const auto id = extract_u64_field(line, "id");
        const auto op_opt = extract_string_field(line, "op");
        if (!lsn.has_value() || !id.has_value() || !op_opt.has_value()) {
            return std::nullopt;
        }
        WalRecord rec;
        rec.lsn = *lsn;
        rec.id = *id;
        rec.op = *op_opt;
        if (rec.op == "INSERT") {
            const auto meta = extract_string_field(line, "metadata");
            const auto vector_s = extract_string_field(line, "vector");
            if (!meta.has_value() || !vector_s.has_value()) {
                return std::nullopt;
            }
            rec.metadata_json = json_unescape(*meta);
            const auto decoded = decode_vector_csv(json_unescape(*vector_s));
            if (!decoded.has_value()) {
                return std::nullopt;
            }
            rec.vector_fp32 = *decoded;
        } else if (rec.op == "UPDATE_META") {
            const auto meta = extract_string_field(line, "metadata");
            if (!meta.has_value()) {
                return std::nullopt;
            }
            rec.metadata_json = json_unescape(*meta);
        } else if (rec.op != "DELETE") {
            return std::nullopt;
        }
        return rec;
    }

    Status load_wal_records(std::vector<WalRecord>* out_records) {
        out_records->clear();
        const fs::path p = wal_path();
        if (!fs::exists(p)) {
            return Status::Ok();
        }
        std::ifstream in(p, std::ios::binary);
        if (!in) {
            return Status::Error("failed opening wal file: " + p.string());
        }
        std::string line;
        std::size_t line_no = 0;
        while (std::getline(in, line)) {
            ++line_no;
            if (trim_copy(line).empty()) {
                continue;
            }
            auto parsed = parse_wal_line(line);
            if (!parsed.has_value()) {
                if (in.eof()) {
                    break;
                }
                return Status::Error("failed parsing wal line " + std::to_string(line_no));
            }
            out_records->push_back(*parsed);
            if (parsed->lsn > last_lsn) {
                last_lsn = parsed->lsn;
            }
        }
        return Status::Ok();
    }

    Status load_ids_and_vectors() {
        total_rows = 0;
        entries.clear();

        const fs::path ids = seg_ids(active_segment_id);
        if (!fs::exists(ids)) {
            return Status::Ok();
        }

        std::ifstream ids_in(ids, std::ios::binary);
        if (!ids_in) {
            return Status::Error("failed to open ids file: " + ids.string());
        }

        std::size_t row = 0;
        while (true) {
            std::uint64_t id = 0;
            ids_in.read(reinterpret_cast<char*>(&id), sizeof(id));
            if (ids_in.eof()) {
                break;
            }
            if (!ids_in.good()) {
                return Status::Error("failed reading ids file: " + ids.string());
            }
            entries[id] = Entry{id, row, false, "{}"};
            ++row;
        }
        total_rows = row;
        return Status::Ok();
    }

    Status load_metadata() {
        const fs::path meta = seg_meta(active_segment_id);
        if (!fs::exists(meta)) {
            return Status::Ok();
        }
        std::ifstream in(meta, std::ios::binary);
        if (!in) {
            return Status::Error("failed opening metadata file: " + meta.string());
        }
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            const auto id_pos = line.find("\"id\":");
            const auto meta_pos = line.find("\"metadata\":");
            if (id_pos == std::string::npos || meta_pos == std::string::npos) {
                continue;
            }
            const auto id_start = line.find_first_of("0123456789", id_pos);
            if (id_start == std::string::npos) {
                continue;
            }
            const auto id_end = line.find_first_not_of("0123456789", id_start);
            const auto id_str = line.substr(id_start, id_end - id_start);
            std::uint64_t id = 0;
            try {
                id = static_cast<std::uint64_t>(std::stoull(id_str));
            } catch (...) {
                continue;
            }
            const auto q1 = line.find('"', meta_pos + 11);
            if (q1 == std::string::npos) {
                continue;
            }
            const auto q2 = line.find_last_of('"');
            if (q2 == std::string::npos || q2 <= q1) {
                continue;
            }
            const auto escaped_meta = line.substr(q1 + 1, q2 - q1 - 1);
            auto it = entries.find(id);
            if (it != entries.end()) {
                it->second.metadata_json = json_unescape(escaped_meta);
            }
        }
        return Status::Ok();
    }

    Status load_tombstones() {
        const fs::path tomb = seg_tomb(active_segment_id);
        if (!fs::exists(tomb)) {
            return Status::Ok();
        }
        std::ifstream in(tomb, std::ios::binary);
        if (!in) {
            return Status::Error("failed opening tombstone file: " + tomb.string());
        }
        while (true) {
            std::uint64_t id = 0;
            in.read(reinterpret_cast<char*>(&id), sizeof(id));
            if (in.eof()) {
                break;
            }
            if (!in.good()) {
                return Status::Error("failed reading tombstone file: " + tomb.string());
            }
            auto it = entries.find(id);
            if (it != entries.end()) {
                it->second.deleted = true;
            }
        }
        return Status::Ok();
    }

    Status load_manifest() {
        const fs::path manifest = manifest_path();
        if (!fs::exists(manifest)) {
            return Status::Ok();
        }
        std::ifstream in(manifest, std::ios::binary);
        if (!in) {
            return Status::Error("failed opening manifest: " + manifest.string());
        }
        std::ostringstream os;
        os << in.rdbuf();
        const std::string text = os.str();

        const auto key = text.find("\"active_segment_id\"");
        if (key == std::string::npos) {
            return Status::Ok();
        }
        const auto n0 = text.find_first_of("0123456789", key);
        if (n0 == std::string::npos) {
            return Status::Ok();
        }
        const auto n1 = text.find_first_not_of("0123456789", n0);
        try {
            active_segment_id = static_cast<std::uint64_t>(std::stoull(text.substr(n0, n1 - n0)));
        } catch (...) {
            return Status::Error("invalid active_segment_id in manifest");
        }
        if (const auto checkpoint = extract_u64_field(text, "checkpoint_lsn"); checkpoint.has_value()) {
            checkpoint_lsn = *checkpoint;
            if (last_lsn < checkpoint_lsn) {
                last_lsn = checkpoint_lsn;
            }
        }
        return Status::Ok();
    }

    Status load_dirty_ranges() {
        dirty_ranges.clear();
        const fs::path p = dirty_ranges_path();
        if (!fs::exists(p)) {
            return Status::Ok();
        }
        std::ifstream in(p, std::ios::binary);
        if (!in) {
            return Status::Error("failed opening dirty ranges file: " + p.string());
        }
        std::ostringstream os;
        os << in.rdbuf();
        const std::string text = os.str();

        std::size_t pos = 0;
        while (true) {
            const auto seg_k = text.find("\"segment_id\"", pos);
            if (seg_k == std::string::npos) {
                break;
            }
            const auto start_k = text.find("\"start_row\"", seg_k);
            const auto end_k = text.find("\"end_row\"", start_k);
            const auto reason_k = text.find("\"reason\"", end_k);
            if (start_k == std::string::npos || end_k == std::string::npos || reason_k == std::string::npos) {
                break;
            }

            auto parse_u64 = [&](std::size_t start_at) -> std::optional<std::uint64_t> {
                const auto n0 = text.find_first_of("0123456789", start_at);
                if (n0 == std::string::npos) {
                    return std::nullopt;
                }
                const auto n1 = text.find_first_not_of("0123456789", n0);
                try {
                    return static_cast<std::uint64_t>(std::stoull(text.substr(n0, n1 - n0)));
                } catch (...) {
                    return std::nullopt;
                }
            };

            const auto seg_v = parse_u64(seg_k);
            const auto start_v = parse_u64(start_k);
            const auto end_v = parse_u64(end_k);
            const auto q1 = text.find('"', reason_k + 8);
            const auto q2 = (q1 == std::string::npos) ? std::string::npos : text.find('"', q1 + 1);

            if (!seg_v.has_value() || !start_v.has_value() || !end_v.has_value() || q1 == std::string::npos || q2 == std::string::npos) {
                pos = reason_k + 1;
                continue;
            }

            dirty_ranges.push_back(DirtyRange{
                *seg_v,
                static_cast<std::size_t>(*start_v),
                static_cast<std::size_t>(*end_v),
                json_unescape(text.substr(q1 + 1, q2 - q1 - 1)),
            });
            pos = q2 + 1;
        }
        return Status::Ok();
    }

    InitialClusteringConfig make_clustering_config(std::uint32_t seed) const {
        InitialClusteringConfig cfg;
        cfg.seed = seed;
        cfg.contiguous_live_vector_load_enabled = env_flag_enabled(
            "VECTOR_DB_CONTIGUOUS_LOAD",
            cfg.contiguous_live_vector_load_enabled);
        cfg.contiguous_min_span_rows = env_size_value(
            "VECTOR_DB_CONTIGUOUS_MIN_SPAN",
            cfg.contiguous_min_span_rows);
        cfg.async_double_buffer_enabled = env_flag_enabled(
            "VECTOR_DB_ASYNC_DOUBLE_BUFFER",
            cfg.async_double_buffer_enabled);
        cfg.async_double_buffer_chunk_rows = env_size_value(
            "VECTOR_DB_ASYNC_CHUNK_ROWS",
            cfg.async_double_buffer_chunk_rows);
        cfg.async_double_buffer_queue_depth = env_size_value(
            "VECTOR_DB_ASYNC_QUEUE_DEPTH",
            cfg.async_double_buffer_queue_depth);
        cfg.elbow_stage_a_approx_enabled = env_flag_enabled(
            "VECTOR_DB_ELBOW_APPROX_STAGE_A",
            cfg.elbow_stage_a_approx_enabled);
        cfg.elbow_stage_a_approx_stride = env_size_value(
            "VECTOR_DB_ELBOW_APPROX_STRIDE",
            cfg.elbow_stage_a_approx_stride);
        cfg.elbow_prune_enabled = env_flag_enabled(
            "VECTOR_DB_ELBOW_PRUNE",
            cfg.elbow_prune_enabled);
        const char* prune_margin_env = std::getenv("VECTOR_DB_ELBOW_PRUNE_MARGIN");
        if (prune_margin_env != nullptr) {
            try {
                cfg.elbow_prune_margin = std::stod(prune_margin_env);
            } catch (...) {
            }
        }
        cfg.elbow_trace_full_grid = env_flag_enabled(
            "VECTOR_DB_ELBOW_TRACE_FULL_GRID",
            cfg.elbow_trace_full_grid);
        cfg.elbow_int8_search_enabled = true;
        cfg.elbow_int8_require_hardware = true;
        const char* int8_scale_mode_env = std::getenv("VECTOR_DB_ELBOW_INT8_SCALE_MODE");
        if (int8_scale_mode_env != nullptr && std::string(int8_scale_mode_env).size() > 0) {
            cfg.elbow_int8_scale_mode = int8_scale_mode_env;
        }
        return cfg;
    }

    LiveVectorLoadResult collect_live_vectors(const InitialClusteringConfig& cfg) const {
        LiveVectorLoadResult out;
        out.vectors_by_id.reserve(entries.size());
        std::vector<std::pair<std::size_t, std::uint64_t>> live_rows;
        live_rows.reserve(entries.size());
        for (const auto& kv : entries) {
            if (!kv.second.deleted) {
                live_rows.push_back({kv.second.row, kv.first});
            }
        }
        if (live_rows.empty()) {
            return out;
        }
        std::sort(live_rows.begin(), live_rows.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        const fs::path vec_path = seg_vec(active_segment_id);
        std::ifstream vec_in(vec_path, std::ios::binary);
        if (!vec_in) {
            return out;
        }
        struct Span {
            std::size_t begin = 0;
            std::size_t end = 0;
        };
        std::vector<Span> spans;
        spans.reserve(live_rows.size());
        std::size_t span_begin = 0;
        for (std::size_t i = 1; i <= live_rows.size(); ++i) {
            const bool contiguous =
                (i < live_rows.size()) && (live_rows[i].first == live_rows[i - 1].first + 1);
            if (!contiguous) {
                spans.push_back(Span{span_begin, i - 1});
                span_begin = i;
            }
        }
        out.contiguous_spans = spans.size();

        const auto read_sparse_row = [&](std::size_t row, std::vector<float>* dst) -> bool {
            const std::uint64_t byte_off = static_cast<std::uint64_t>(row)
                * static_cast<std::uint64_t>(kVectorDim) * sizeof(float);
            vec_in.seekg(static_cast<std::streamoff>(byte_off), std::ios::beg);
            if (!vec_in.good()) {
                vec_in.clear();
                return false;
            }
            vec_in.read(
                reinterpret_cast<char*>(dst->data()),
                static_cast<std::streamsize>(kVectorDim * sizeof(float)));
            if (!vec_in.good()) {
                vec_in.clear();
                return false;
            }
            out.bytes_read += kVectorDim * sizeof(float);
            out.sparse_reads += 1;
            return true;
        };

        const bool contiguous_enabled = cfg.contiguous_live_vector_load_enabled;
        const std::size_t min_span_rows = std::max<std::size_t>(2, cfg.contiguous_min_span_rows);

        if (!contiguous_enabled) {
            std::vector<float> buf(kVectorDim, 0.0f);
            for (const auto& [row, id] : live_rows) {
                if (read_sparse_row(row, &buf)) {
                    out.vectors_by_id.push_back({id, buf});
                }
            }
        } else if (cfg.async_double_buffer_enabled) {
            out.async_double_buffer_used = true;
            struct Chunk {
                std::size_t start = 0;
                std::size_t rows = 0;
                std::vector<float> data;
                bool ok = true;
            };
            std::deque<Chunk> queue;
            std::mutex mu;
            std::condition_variable cv_push;
            std::condition_variable cv_pop;
            bool done = false;
            const std::size_t max_queue = std::max<std::size_t>(1, cfg.async_double_buffer_queue_depth);
            std::string producer_error;

            std::thread producer([&]() {
                std::ifstream pin(vec_path, std::ios::binary);
                if (!pin) {
                    producer_error = "failed opening vector file for async live load: " + vec_path.string();
                    done = true;
                    cv_pop.notify_all();
                    return;
                }
                for (const Span& sp : spans) {
                    const std::size_t span_rows = sp.end - sp.begin + 1;
                    if (span_rows < min_span_rows) {
                        continue;
                    }
                    const std::size_t chunk_rows =
                        std::max<std::size_t>(1, cfg.async_double_buffer_chunk_rows);
                    for (std::size_t offset = 0; offset < span_rows; offset += chunk_rows) {
                        const std::size_t rows_this = std::min(chunk_rows, span_rows - offset);
                        const std::size_t start_idx = sp.begin + offset;
                        const std::size_t row0 = live_rows[start_idx].first;
                        const std::uint64_t byte_off = static_cast<std::uint64_t>(row0)
                            * static_cast<std::uint64_t>(kVectorDim) * sizeof(float);
                        Chunk c{};
                        c.start = start_idx;
                        c.rows = rows_this;
                        c.data.resize(rows_this * kVectorDim, 0.0f);
                        pin.seekg(static_cast<std::streamoff>(byte_off), std::ios::beg);
                        if (!pin.good()) {
                            pin.clear();
                            c.ok = false;
                        } else {
                            pin.read(
                                reinterpret_cast<char*>(c.data.data()),
                                static_cast<std::streamsize>(c.data.size() * sizeof(float)));
                            if (!pin.good()) {
                                pin.clear();
                                c.ok = false;
                            }
                        }
                        std::unique_lock<std::mutex> lk(mu);
                        cv_push.wait(lk, [&]() { return queue.size() < max_queue; });
                        queue.push_back(std::move(c));
                        lk.unlock();
                        cv_pop.notify_one();
                    }
                }
                std::unique_lock<std::mutex> lk(mu);
                done = true;
                lk.unlock();
                cv_pop.notify_all();
            });

            for (;;) {
                Chunk c;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv_pop.wait(lk, [&]() { return done || !queue.empty(); });
                    if (queue.empty()) {
                        break;
                    }
                    c = std::move(queue.front());
                    queue.pop_front();
                    lk.unlock();
                    cv_push.notify_one();
                }
                if (c.ok) {
                    out.bytes_read += c.data.size() * sizeof(float);
                    for (std::size_t i = 0; i < c.rows; ++i) {
                        const std::size_t row_idx = c.start + i;
                        const std::uint64_t id = live_rows[row_idx].second;
                        std::vector<float> v(kVectorDim, 0.0f);
                        std::copy_n(
                            c.data.data() + static_cast<std::ptrdiff_t>(i * kVectorDim),
                            kVectorDim,
                            v.data());
                        out.vectors_by_id.push_back({id, std::move(v)});
                    }
                } else {
                    out.sparse_fallback_used = true;
                    std::vector<float> buf(kVectorDim, 0.0f);
                    for (std::size_t i = 0; i < c.rows; ++i) {
                        const std::size_t row_idx = c.start + i;
                        if (read_sparse_row(live_rows[row_idx].first, &buf)) {
                            out.vectors_by_id.push_back({live_rows[row_idx].second, buf});
                        }
                    }
                }
            }
            producer.join();
            if (!producer_error.empty()) {
                out.vectors_by_id.clear();
                out.packed_row_major.clear();
                return out;
            }
            // Any short span rows are processed via sparse fallback.
            std::vector<float> buf(kVectorDim, 0.0f);
            for (const Span& sp : spans) {
                const std::size_t span_rows = sp.end - sp.begin + 1;
                if (span_rows >= min_span_rows) {
                    continue;
                }
                out.sparse_fallback_used = true;
                for (std::size_t i = sp.begin; i <= sp.end; ++i) {
                    if (read_sparse_row(live_rows[i].first, &buf)) {
                        out.vectors_by_id.push_back({live_rows[i].second, buf});
                    }
                }
            }
        } else {
            std::vector<float> buf(kVectorDim, 0.0f);
            for (const Span& sp : spans) {
                const std::size_t span_rows = sp.end - sp.begin + 1;
                if (span_rows < min_span_rows) {
                    out.sparse_fallback_used = true;
                    for (std::size_t i = sp.begin; i <= sp.end; ++i) {
                        if (read_sparse_row(live_rows[i].first, &buf)) {
                            out.vectors_by_id.push_back({live_rows[i].second, buf});
                        }
                    }
                    continue;
                }
                const std::size_t start_row = live_rows[sp.begin].first;
                const std::uint64_t byte_off = static_cast<std::uint64_t>(start_row)
                    * static_cast<std::uint64_t>(kVectorDim) * sizeof(float);
                std::vector<float> chunk(span_rows * kVectorDim, 0.0f);
                vec_in.seekg(static_cast<std::streamoff>(byte_off), std::ios::beg);
                if (!vec_in.good()) {
                    vec_in.clear();
                    out.sparse_fallback_used = true;
                    for (std::size_t i = sp.begin; i <= sp.end; ++i) {
                        if (read_sparse_row(live_rows[i].first, &buf)) {
                            out.vectors_by_id.push_back({live_rows[i].second, buf});
                        }
                    }
                    continue;
                }
                vec_in.read(
                    reinterpret_cast<char*>(chunk.data()),
                    static_cast<std::streamsize>(chunk.size() * sizeof(float)));
                if (!vec_in.good()) {
                    vec_in.clear();
                    out.sparse_fallback_used = true;
                    for (std::size_t i = sp.begin; i <= sp.end; ++i) {
                        if (read_sparse_row(live_rows[i].first, &buf)) {
                            out.vectors_by_id.push_back({live_rows[i].second, buf});
                        }
                    }
                    continue;
                }
                out.bytes_read += chunk.size() * sizeof(float);
                for (std::size_t i = 0; i < span_rows; ++i) {
                    const std::uint64_t id = live_rows[sp.begin + i].second;
                    std::vector<float> v(kVectorDim, 0.0f);
                    std::copy_n(
                        chunk.data() + static_cast<std::ptrdiff_t>(i * kVectorDim),
                        kVectorDim,
                        v.data());
                    out.vectors_by_id.push_back({id, std::move(v)});
                }
            }
        }

        std::sort(out.vectors_by_id.begin(), out.vectors_by_id.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        out.packed_row_major.reserve(out.vectors_by_id.size() * kVectorDim);
        for (const auto& kv : out.vectors_by_id) {
            out.packed_row_major.insert(
                out.packed_row_major.end(),
                kv.second.begin(),
                kv.second.end());
        }
        return out;
    }

    Status collect_vectors_for_ids(
        const std::vector<std::uint64_t>& ids,
        LiveVectorLoadResult* out) const {
        if (out == nullptr) {
            return Status::Error("collect_vectors_for_ids output is null");
        }
        out->vectors_by_id.clear();
        out->packed_row_major.clear();
        out->bytes_read = 0;
        out->contiguous_spans = 0;
        out->sparse_reads = 0;
        out->sparse_fallback_used = false;
        out->async_double_buffer_used = false;
        out->vectors_by_id.reserve(ids.size());
        for (std::uint64_t id : ids) {
            const auto it = entries.find(id);
            if (it == entries.end() || it->second.deleted) {
                continue;
            }
            const auto vec = read_vector_at_row(it->second.row);
            if (!vec.has_value()) {
                continue;
            }
            out->vectors_by_id.push_back({id, *vec});
            out->bytes_read += kVectorDim * sizeof(float);
            out->sparse_reads += 1;
        }
        std::sort(out->vectors_by_id.begin(), out->vectors_by_id.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        out->packed_row_major.reserve(out->vectors_by_id.size() * kVectorDim);
        for (const auto& kv : out->vectors_by_id) {
            out->packed_row_major.insert(
                out->packed_row_major.end(),
                kv.second.begin(),
                kv.second.end());
        }
        return Status::Ok();
    }

    Status load_top1_assignment_groups(
        std::uint64_t source_version,
        std::unordered_map<std::uint32_t, std::vector<std::uint64_t>>* out_groups) const {
        if (out_groups == nullptr) {
            return Status::Error("load_top1_assignment_groups output is null");
        }
        out_groups->clear();
        const fs::path p = cluster_version_dir(source_version) / "assignments.json";
        if (!fs::exists(p)) {
            return Status::Error("missing first-layer assignments file: " + p.string());
        }
        std::ifstream in(p, std::ios::binary);
        if (!in) {
            return Status::Error("failed opening first-layer assignments file: " + p.string());
        }
        std::ostringstream os;
        os << in.rdbuf();
        const std::string text = os.str();
        std::size_t pos = 0;
        while (true) {
            const auto id_k = text.find("\"id\"", pos);
            if (id_k == std::string::npos) {
                break;
            }
            const auto top_k = text.find("\"top\"", id_k);
            if (top_k == std::string::npos) {
                break;
            }
            const auto id_v = extract_u64_field(text.substr(id_k), "id");
            if (!id_v.has_value()) {
                pos = top_k + 1;
                continue;
            }
            const auto lb = text.find('[', top_k);
            const auto rb = (lb == std::string::npos) ? std::string::npos : text.find(']', lb);
            if (lb == std::string::npos || rb == std::string::npos) {
                pos = top_k + 1;
                continue;
            }
            const auto n0 = text.find_first_of("0123456789", lb);
            if (n0 == std::string::npos || n0 > rb) {
                pos = rb + 1;
                continue;
            }
            const auto n1 = text.find_first_not_of("0123456789", n0);
            std::uint32_t centroid = 0;
            try {
                centroid = static_cast<std::uint32_t>(
                    std::stoul(text.substr(n0, n1 - n0)));
            } catch (...) {
                pos = rb + 1;
                continue;
            }
            (*out_groups)[centroid].push_back(*id_v);
            pos = rb + 1;
        }
        if (out_groups->empty()) {
            return Status::Error("first-layer assignments has no parseable top-1 groups");
        }
        return Status::Ok();
    }

    std::uint64_t next_cluster_version() const {
        return cluster_stats_cache.available ? (cluster_stats_cache.version + 1) : 1;
    }

    Status load_cluster_manifest() {
        cluster_stats_cache = ClusterStats{};
        cluster_health_cache = ClusterHealth{};
        const fs::path p = cluster_manifest_path();
        if (!fs::exists(p)) {
            return Status::Ok();
        }
        std::ifstream in(p, std::ios::binary);
        if (!in) {
            return Status::Error("failed opening cluster manifest: " + p.string());
        }
        std::ostringstream os;
        os << in.rdbuf();
        const std::string text = os.str();
        cluster_stats_cache.available = true;
        cluster_health_cache.available = true;
        if (const auto v = extract_u64_field(text, "version"); v.has_value()) {
            cluster_stats_cache.version = *v;
        }
        if (const auto v = extract_u64_field(text, "build_lsn"); v.has_value()) {
            cluster_stats_cache.build_lsn = *v;
        }
        if (const auto v = extract_u64_field(text, "vectors_indexed"); v.has_value()) {
            cluster_stats_cache.vectors_indexed = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_u64_field(text, "chosen_k"); v.has_value()) {
            cluster_stats_cache.chosen_k = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_u64_field(text, "k_min"); v.has_value()) {
            cluster_stats_cache.k_min = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_u64_field(text, "k_max"); v.has_value()) {
            cluster_stats_cache.k_max = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_double_field(text, "objective"); v.has_value()) {
            cluster_stats_cache.objective = *v;
        }
        if (const auto v = extract_bool_field(text, "used_cuda"); v.has_value()) {
            cluster_stats_cache.used_cuda = *v;
        }
        if (const auto v = extract_bool_field(text, "tensor_core_enabled"); v.has_value()) {
            cluster_stats_cache.tensor_core_enabled = *v;
        }
        if (const auto v = extract_string_field(text, "gpu_backend"); v.has_value()) {
            cluster_stats_cache.gpu_backend = *v;
        }
        if (const auto v = extract_double_field(text, "scoring_ms_total"); v.has_value()) {
            cluster_stats_cache.scoring_ms_total = *v;
        }
        if (const auto v = extract_u64_field(text, "scoring_calls"); v.has_value()) {
            cluster_stats_cache.scoring_calls = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_u64_field(text, "elbow_k_evaluated_count"); v.has_value()) {
            cluster_stats_cache.elbow_k_evaluated_count = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_u64_field(text, "elbow_stage_a_candidates"); v.has_value()) {
            cluster_stats_cache.elbow_stage_a_candidates = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_u64_field(text, "elbow_stage_b_candidates"); v.has_value()) {
            cluster_stats_cache.elbow_stage_b_candidates = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_string_field(text, "elbow_early_stop_reason"); v.has_value()) {
            cluster_stats_cache.elbow_early_stop_reason = *v;
        }
        if (const auto v = extract_u64_field(text, "stability_runs_executed"); v.has_value()) {
            cluster_stats_cache.stability_runs_executed = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_double_field(text, "load_live_vectors_ms"); v.has_value()) {
            cluster_stats_cache.load_live_vectors_ms = *v;
        }
        if (const auto v = extract_double_field(text, "id_estimation_ms"); v.has_value()) {
            cluster_stats_cache.id_estimation_ms = *v;
        }
        if (const auto v = extract_double_field(text, "elbow_ms"); v.has_value()) {
            cluster_stats_cache.elbow_ms = *v;
        }
        if (const auto v = extract_double_field(text, "stability_ms"); v.has_value()) {
            cluster_stats_cache.stability_ms = *v;
        }
        if (const auto v = extract_double_field(text, "write_artifacts_ms"); v.has_value()) {
            cluster_stats_cache.write_artifacts_ms = *v;
        }
        if (const auto v = extract_double_field(text, "total_build_ms"); v.has_value()) {
            cluster_stats_cache.total_build_ms = *v;
        }
        if (const auto v = extract_u64_field(text, "live_vector_bytes_read"); v.has_value()) {
            cluster_stats_cache.live_vector_bytes_read = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_u64_field(text, "live_vector_contiguous_spans"); v.has_value()) {
            cluster_stats_cache.live_vector_contiguous_spans = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_u64_field(text, "live_vector_sparse_reads"); v.has_value()) {
            cluster_stats_cache.live_vector_sparse_reads = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_bool_field(text, "live_vector_sparse_fallback"); v.has_value()) {
            cluster_stats_cache.live_vector_sparse_fallback = *v;
        }
        if (const auto v = extract_bool_field(text, "live_vector_async_double_buffer"); v.has_value()) {
            cluster_stats_cache.live_vector_async_double_buffer = *v;
        }
        if (const auto v = extract_bool_field(text, "elbow_stage_a_approx_enabled"); v.has_value()) {
            cluster_stats_cache.elbow_stage_a_approx_enabled = *v;
        }
        if (const auto v = extract_u64_field(text, "elbow_stage_a_approx_dim"); v.has_value()) {
            cluster_stats_cache.elbow_stage_a_approx_dim = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_u64_field(text, "elbow_stage_a_approx_stride"); v.has_value()) {
            cluster_stats_cache.elbow_stage_a_approx_stride = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_u64_field(text, "elbow_stage_b_pruned_candidates"); v.has_value()) {
            cluster_stats_cache.elbow_stage_b_pruned_candidates = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_u64_field(text, "elbow_stage_b_window_k_min"); v.has_value()) {
            cluster_stats_cache.elbow_stage_b_window_k_min = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_u64_field(text, "elbow_stage_b_window_k_max"); v.has_value()) {
            cluster_stats_cache.elbow_stage_b_window_k_max = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_string_field(text, "elbow_stage_b_prune_reason"); v.has_value()) {
            cluster_stats_cache.elbow_stage_b_prune_reason = *v;
        }
        if (const auto v = extract_bool_field(text, "elbow_int8_search_enabled"); v.has_value()) {
            cluster_stats_cache.elbow_int8_search_enabled = *v;
        }
        if (const auto v = extract_bool_field(text, "elbow_int8_tensor_core_used"); v.has_value()) {
            cluster_stats_cache.elbow_int8_tensor_core_used = *v;
        }
        if (const auto v = extract_u64_field(text, "elbow_int8_eval_count"); v.has_value()) {
            cluster_stats_cache.elbow_int8_eval_count = static_cast<std::size_t>(*v);
        }
        if (const auto v = extract_string_field(text, "elbow_int8_scale_mode"); v.has_value()) {
            cluster_stats_cache.elbow_int8_scale_mode = *v;
        }
        if (const auto v = extract_string_field(text, "elbow_scoring_precision"); v.has_value()) {
            cluster_stats_cache.elbow_scoring_precision = *v;
        }
        if (const auto v = extract_double_field(text, "mean_nmi"); v.has_value()) {
            cluster_health_cache.mean_nmi = *v;
        }
        if (const auto v = extract_double_field(text, "std_nmi"); v.has_value()) {
            cluster_health_cache.std_nmi = *v;
        }
        if (const auto v = extract_double_field(text, "mean_jaccard"); v.has_value()) {
            cluster_health_cache.mean_jaccard = *v;
        }
        if (const auto v = extract_double_field(text, "mean_centroid_drift"); v.has_value()) {
            cluster_health_cache.mean_centroid_drift = *v;
        }
        if (const auto v = extract_bool_field(text, "stability_passed"); v.has_value()) {
            cluster_health_cache.passed = *v;
        }
        cluster_health_cache.status = cluster_health_cache.passed ? "ok" : "failed";
        return Status::Ok();
    }

    Status write_cluster_manifest(
        const ClusterStats& stats,
        const ClusterHealth& health,
        const IdEstimateRange& idr,
        bool elbow_fallback) const {
        std::ostringstream os;
        os << "{\n";
        os << "  \"version\": " << stats.version << ",\n";
        os << "  \"build_lsn\": " << stats.build_lsn << ",\n";
        os << "  \"vectors_indexed\": " << stats.vectors_indexed << ",\n";
        os << "  \"chosen_k\": " << stats.chosen_k << ",\n";
        os << "  \"k_min\": " << stats.k_min << ",\n";
        os << "  \"k_max\": " << stats.k_max << ",\n";
        os << "  \"objective\": " << std::setprecision(10) << stats.objective << ",\n";
        os << "  \"used_cuda\": " << (stats.used_cuda ? "true" : "false") << ",\n";
        os << "  \"tensor_core_enabled\": " << (stats.tensor_core_enabled ? "true" : "false") << ",\n";
        os << "  \"gpu_backend\": \"" << json_escape(stats.gpu_backend) << "\",\n";
        os << "  \"scoring_ms_total\": " << stats.scoring_ms_total << ",\n";
        os << "  \"scoring_calls\": " << stats.scoring_calls << ",\n";
        os << "  \"elbow_k_evaluated_count\": " << stats.elbow_k_evaluated_count << ",\n";
        os << "  \"elbow_stage_a_candidates\": " << stats.elbow_stage_a_candidates << ",\n";
        os << "  \"elbow_stage_b_candidates\": " << stats.elbow_stage_b_candidates << ",\n";
        os << "  \"elbow_early_stop_reason\": \"" << json_escape(stats.elbow_early_stop_reason) << "\",\n";
        os << "  \"stability_runs_executed\": " << stats.stability_runs_executed << ",\n";
        os << "  \"load_live_vectors_ms\": " << stats.load_live_vectors_ms << ",\n";
        os << "  \"id_estimation_ms\": " << stats.id_estimation_ms << ",\n";
        os << "  \"elbow_ms\": " << stats.elbow_ms << ",\n";
        os << "  \"stability_ms\": " << stats.stability_ms << ",\n";
        os << "  \"write_artifacts_ms\": " << stats.write_artifacts_ms << ",\n";
        os << "  \"total_build_ms\": " << stats.total_build_ms << ",\n";
        os << "  \"live_vector_bytes_read\": " << stats.live_vector_bytes_read << ",\n";
        os << "  \"live_vector_contiguous_spans\": " << stats.live_vector_contiguous_spans << ",\n";
        os << "  \"live_vector_sparse_reads\": " << stats.live_vector_sparse_reads << ",\n";
        os << "  \"live_vector_sparse_fallback\": "
           << (stats.live_vector_sparse_fallback ? "true" : "false") << ",\n";
        os << "  \"live_vector_async_double_buffer\": "
           << (stats.live_vector_async_double_buffer ? "true" : "false") << ",\n";
        os << "  \"elbow_stage_a_approx_enabled\": "
           << (stats.elbow_stage_a_approx_enabled ? "true" : "false") << ",\n";
        os << "  \"elbow_stage_a_approx_dim\": " << stats.elbow_stage_a_approx_dim << ",\n";
        os << "  \"elbow_stage_a_approx_stride\": " << stats.elbow_stage_a_approx_stride << ",\n";
        os << "  \"elbow_stage_b_pruned_candidates\": " << stats.elbow_stage_b_pruned_candidates << ",\n";
        os << "  \"elbow_stage_b_window_k_min\": " << stats.elbow_stage_b_window_k_min << ",\n";
        os << "  \"elbow_stage_b_window_k_max\": " << stats.elbow_stage_b_window_k_max << ",\n";
        os << "  \"elbow_stage_b_prune_reason\": \"" << json_escape(stats.elbow_stage_b_prune_reason)
           << "\",\n";
        os << "  \"elbow_int8_search_enabled\": "
           << (stats.elbow_int8_search_enabled ? "true" : "false") << ",\n";
        os << "  \"elbow_int8_tensor_core_used\": "
           << (stats.elbow_int8_tensor_core_used ? "true" : "false") << ",\n";
        os << "  \"elbow_int8_eval_count\": " << stats.elbow_int8_eval_count << ",\n";
        os << "  \"elbow_int8_scale_mode\": \"" << json_escape(stats.elbow_int8_scale_mode)
           << "\",\n";
        os << "  \"elbow_scoring_precision\": \"" << json_escape(stats.elbow_scoring_precision)
           << "\",\n";
        os << "  \"id_sample_size\": " << idr.sample_size << ",\n";
        os << "  \"id_m_low\": " << idr.m_low << ",\n";
        os << "  \"id_m_high\": " << idr.m_high << ",\n";
        os << "  \"elbow_fallback\": " << (elbow_fallback ? "true" : "false") << ",\n";
        os << "  \"mean_nmi\": " << health.mean_nmi << ",\n";
        os << "  \"std_nmi\": " << health.std_nmi << ",\n";
        os << "  \"mean_jaccard\": " << health.mean_jaccard << ",\n";
        os << "  \"mean_centroid_drift\": " << health.mean_centroid_drift << ",\n";
        os << "  \"stability_passed\": " << (health.passed ? "true" : "false") << "\n";
        os << "}\n";
        return write_text_atomic(cluster_manifest_path(), os.str());
    }

    Status write_cluster_artifacts_to_dir(
        const fs::path& out_dir,
        const IdEstimateRange& idr,
        const ElbowSelection& elbow,
        const KMeansModel& model,
        const StabilityMetrics& stability,
        const std::vector<std::pair<std::uint64_t, std::vector<float>>>& vectors) const {
        std::error_code ec;
        fs::create_directories(out_dir, ec);
        if (ec) {
            return Status::Error("failed creating clustering output directory: " + out_dir.string());
        }

        {
            std::ostringstream os;
            os << "{\n";
            os << "  \"sample_size\": " << idr.sample_size << ",\n";
            os << "  \"m_low\": " << idr.m_low << ",\n";
            os << "  \"m_high\": " << idr.m_high << ",\n";
            os << "  \"k_min\": " << idr.k_min << ",\n";
            os << "  \"k_max\": " << idr.k_max << "\n";
            os << "}\n";
            if (const Status s = write_text_atomic(out_dir / "id_estimate.json", os.str()); !s.ok) {
                return s;
            }
        }
        {
            std::ostringstream os;
            os << "{\n";
            os << "  \"chosen_k\": " << elbow.chosen_k << ",\n";
            os << "  \"used_fallback\": " << (elbow.used_fallback ? "true" : "false") << ",\n";
            os << "  \"trace\": [\n";
            for (std::size_t i = 0; i < elbow.trace.size(); ++i) {
                const auto& t = elbow.trace[i];
                os << "    {\"k\": " << t.k << ", \"objective\": " << t.objective << ", \"gain_to_2k\": " << t.gain_to_2k << "}";
                if (i + 1 < elbow.trace.size()) {
                    os << ",";
                }
                os << "\n";
            }
            os << "  ]\n}\n";
            if (const Status s = write_text_atomic(out_dir / "elbow_trace.json", os.str()); !s.ok) {
                return s;
            }
        }
        {
            std::ostringstream os;
            os << "{\n";
            os << "  \"mean_nmi\": " << stability.mean_nmi << ",\n";
            os << "  \"std_nmi\": " << stability.std_nmi << ",\n";
            os << "  \"mean_jaccard\": " << stability.mean_jaccard << ",\n";
            os << "  \"mean_centroid_drift\": " << stability.mean_centroid_drift << ",\n";
            os << "  \"passed\": " << (stability.passed ? "true" : "false") << "\n";
            os << "}\n";
            if (const Status s = write_text_atomic(out_dir / "stability_report.json", os.str()); !s.ok) {
                return s;
            }
        }
        {
            std::ofstream cent_out(out_dir / "centroids.bin", std::ios::binary | std::ios::trunc);
            if (!cent_out) {
                return Status::Error("failed opening centroids.bin for write");
            }
            cent_out.write(
                reinterpret_cast<const char*>(model.centroids.data()),
                static_cast<std::streamsize>(model.centroids.size() * sizeof(float)));
            if (!cent_out.good()) {
                return Status::Error("failed writing centroids.bin");
            }
        }
        {
            std::ostringstream os;
            os << "{\n  \"assignments\": [\n";
            for (std::size_t i = 0; i < vectors.size(); ++i) {
                os << "    {\"id\": " << vectors[i].first << ", \"top\": [";
                for (std::size_t j = 0; j < model.assignments[i].size(); ++j) {
                    if (j > 0) {
                        os << ", ";
                    }
                    os << model.assignments[i][j];
                }
                os << "]}";
                if (i + 1 < vectors.size()) {
                    os << ",";
                }
                os << "\n";
            }
            os << "  ]\n}\n";
            if (const Status s = write_text_atomic(out_dir / "assignments.json", os.str()); !s.ok) {
                return s;
            }
        }
        return Status::Ok();
    }

    Status write_initial_cluster_artifacts(
        std::uint64_t version,
        const IdEstimateRange& idr,
        const ElbowSelection& elbow,
        const KMeansModel& model,
        const StabilityMetrics& stability,
        const std::vector<std::pair<std::uint64_t, std::vector<float>>>& vectors) const {
        const fs::path out_dir = clusters_root() / ("v" + std::to_string(version));
        return write_cluster_artifacts_to_dir(out_dir, idr, elbow, model, stability, vectors);
    }

    Status write_second_level_centroid_manifest(
        const fs::path& out_dir,
        std::uint64_t source_version,
        std::uint32_t centroid_id,
        std::size_t source_vectors,
        const ClusterStats& stats,
        const ClusterHealth& health) const {
        std::ostringstream os;
        os << "{\n";
        os << "  \"source_version\": " << source_version << ",\n";
        os << "  \"parent_centroid_id\": " << centroid_id << ",\n";
        os << "  \"source_vectors\": " << source_vectors << ",\n";
        os << "  \"vectors_indexed\": " << stats.vectors_indexed << ",\n";
        os << "  \"chosen_k\": " << stats.chosen_k << ",\n";
        os << "  \"k_min\": " << stats.k_min << ",\n";
        os << "  \"k_max\": " << stats.k_max << ",\n";
        os << "  \"objective\": " << stats.objective << ",\n";
        os << "  \"used_cuda\": " << (stats.used_cuda ? "true" : "false") << ",\n";
        os << "  \"tensor_core_enabled\": " << (stats.tensor_core_enabled ? "true" : "false") << ",\n";
        os << "  \"gpu_backend\": \"" << json_escape(stats.gpu_backend) << "\",\n";
        os << "  \"scoring_ms_total\": " << stats.scoring_ms_total << ",\n";
        os << "  \"scoring_calls\": " << stats.scoring_calls << ",\n";
        os << "  \"elbow_int8_search_enabled\": "
           << (stats.elbow_int8_search_enabled ? "true" : "false") << ",\n";
        os << "  \"elbow_int8_tensor_core_used\": "
           << (stats.elbow_int8_tensor_core_used ? "true" : "false") << ",\n";
        os << "  \"elbow_int8_eval_count\": " << stats.elbow_int8_eval_count << ",\n";
        os << "  \"elbow_scoring_precision\": \"" << json_escape(stats.elbow_scoring_precision) << "\",\n";
        os << "  \"stability_passed\": " << (health.passed ? "true" : "false") << ",\n";
        os << "  \"mean_nmi\": " << health.mean_nmi << ",\n";
        os << "  \"std_nmi\": " << health.std_nmi << ",\n";
        os << "  \"mean_jaccard\": " << health.mean_jaccard << ",\n";
        os << "  \"mean_centroid_drift\": " << health.mean_centroid_drift << "\n";
        os << "}\n";
        return write_text_atomic(out_dir / "manifest.json", os.str());
    }

    Status write_second_level_summary(
        std::uint64_t source_version,
        std::uint32_t seed,
        const std::vector<SecondLevelCentroidSummary>& summaries,
        double total_elapsed_ms) const {
        const fs::path out_dir = second_level_root(source_version);
        std::error_code ec;
        fs::create_directories(out_dir, ec);
        if (ec) {
            return Status::Error("failed creating second-level output directory: " + out_dir.string());
        }
        std::size_t processed_count = 0;
        std::size_t skipped_count = 0;
        std::size_t total_vectors_processed = 0;
        for (const auto& s : summaries) {
            if (s.processed) {
                ++processed_count;
                total_vectors_processed += s.vectors_indexed;
            } else {
                ++skipped_count;
            }
        }
        const double vectors_per_second =
            (total_elapsed_ms > 0.0)
            ? (static_cast<double>(total_vectors_processed) / (total_elapsed_ms / 1000.0))
            : 0.0;
        std::ostringstream os;
        os << "{\n";
        os << "  \"source_version\": " << source_version << ",\n";
        os << "  \"seed\": " << seed << ",\n";
        os << "  \"total_parent_centroids\": " << summaries.size() << ",\n";
        os << "  \"processed_centroids\": " << processed_count << ",\n";
        os << "  \"skipped_centroids\": " << skipped_count << ",\n";
        os << "  \"total_vectors_processed\": " << total_vectors_processed << ",\n";
        os << "  \"total_elapsed_ms\": " << total_elapsed_ms << ",\n";
        os << "  \"vectors_per_second\": " << vectors_per_second << ",\n";
        os << "  \"centroids\": [\n";
        for (std::size_t i = 0; i < summaries.size(); ++i) {
            const auto& s = summaries[i];
            os << "    {\n";
            os << "      \"centroid_id\": " << s.centroid_id << ",\n";
            os << "      \"source_vectors\": " << s.source_vectors << ",\n";
            os << "      \"vectors_indexed\": " << s.vectors_indexed << ",\n";
            os << "      \"processed\": " << (s.processed ? "true" : "false") << ",\n";
            os << "      \"skipped_reason\": \"" << json_escape(s.skipped_reason) << "\",\n";
            os << "      \"output_path\": \"" << json_escape(s.output_dir.string()) << "\",\n";
            os << "      \"chosen_k\": " << s.stats.chosen_k << ",\n";
            os << "      \"stability_passed\": " << (s.health.passed ? "true" : "false") << ",\n";
            os << "      \"used_cuda\": " << (s.stats.used_cuda ? "true" : "false") << ",\n";
            os << "      \"tensor_core_enabled\": " << (s.stats.tensor_core_enabled ? "true" : "false") << ",\n";
            os << "      \"gpu_backend\": \"" << json_escape(s.stats.gpu_backend) << "\",\n";
            os << "      \"scoring_ms_total\": " << s.stats.scoring_ms_total << ",\n";
            os << "      \"scoring_calls\": " << s.stats.scoring_calls << ",\n";
            os << "      \"elbow_int8_search_enabled\": "
               << (s.stats.elbow_int8_search_enabled ? "true" : "false") << ",\n";
            os << "      \"elbow_int8_tensor_core_used\": "
               << (s.stats.elbow_int8_tensor_core_used ? "true" : "false") << ",\n";
            os << "      \"elbow_int8_eval_count\": " << s.stats.elbow_int8_eval_count << ",\n";
            os << "      \"elbow_scoring_precision\": \""
               << json_escape(s.stats.elbow_scoring_precision) << "\"\n";
            os << "    }";
            if (i + 1 < summaries.size()) {
                os << ",";
            }
            os << "\n";
        }
        os << "  ]\n";
        os << "}\n";
        return write_text_atomic(out_dir / "SECOND_LEVEL_CLUSTERING.json", os.str());
    }

    Status apply_insert_internal(
        std::uint64_t id,
        const std::vector<float>& vector_fp32_1024,
        const std::string& metadata_json,
        bool upsert,
        const std::string& reason) {
        const auto it_existing = entries.find(id);
        if (it_existing != entries.end() && !upsert) {
            return Status::Error("duplicate id rejected");
        }

        if (it_existing != entries.end() && upsert) {
            auto& e = entries[id];
            const fs::path vec = seg_vec(active_segment_id);
            std::fstream vec_io(vec, std::ios::binary | std::ios::in | std::ios::out);
            if (!vec_io) {
                return Status::Error("failed opening vector file for upsert: " + vec.string());
            }
            const std::uint64_t byte_off = static_cast<std::uint64_t>(e.row) * static_cast<std::uint64_t>(kVectorDim) * sizeof(float);
            vec_io.seekp(static_cast<std::streamoff>(byte_off), std::ios::beg);
            vec_io.write(reinterpret_cast<const char*>(vector_fp32_1024.data()), static_cast<std::streamsize>(kVectorDim * sizeof(float)));
            if (!vec_io.good()) {
                return Status::Error("failed writing vector during upsert");
            }
            e.deleted = false;
            e.metadata_json = metadata_json;
            record_dirty(e.row, reason);
            std::ofstream meta_out(seg_meta(active_segment_id), std::ios::binary | std::ios::app);
            if (!meta_out) {
                return Status::Error("failed opening metadata file for upsert");
            }
            meta_out << "{\"id\":" << id << ",\"metadata\":\"" << json_escape(metadata_json) << "\"}\n";
            if (!meta_out.good()) {
                return Status::Error("failed appending metadata during upsert");
            }
            return Status::Ok();
        }

        {
            std::ofstream vec_out(seg_vec(active_segment_id), std::ios::binary | std::ios::app);
            if (!vec_out) {
                return Status::Error("failed opening vector file for append");
            }
            vec_out.write(reinterpret_cast<const char*>(vector_fp32_1024.data()), static_cast<std::streamsize>(kVectorDim * sizeof(float)));
            if (!vec_out.good()) {
                return Status::Error("failed appending vector bytes");
            }
        }
        {
            std::ofstream id_out(seg_ids(active_segment_id), std::ios::binary | std::ios::app);
            if (!id_out) {
                return Status::Error("failed opening ids file for append");
            }
            id_out.write(reinterpret_cast<const char*>(&id), static_cast<std::streamsize>(sizeof(id)));
            if (!id_out.good()) {
                return Status::Error("failed appending id");
            }
        }
        {
            std::ofstream meta_out(seg_meta(active_segment_id), std::ios::binary | std::ios::app);
            if (!meta_out) {
                return Status::Error("failed opening metadata file for append");
            }
            meta_out << "{\"id\":" << id << ",\"metadata\":\"" << json_escape(metadata_json) << "\"}\n";
            if (!meta_out.good()) {
                return Status::Error("failed appending metadata");
            }
        }

        const std::size_t row = total_rows;
        entries[id] = Entry{id, row, false, metadata_json};
        total_rows += 1;
        record_dirty(row, reason);
        return Status::Ok();
    }

    Status apply_remove_internal(std::uint64_t id, const std::string& reason) {
        auto it = entries.find(id);
        if (it == entries.end()) {
            return Status::Error("id not found");
        }
        if (it->second.deleted) {
            return Status::Ok();
        }
        std::ofstream tomb_out(seg_tomb(active_segment_id), std::ios::binary | std::ios::app);
        if (!tomb_out) {
            return Status::Error("failed opening tombstone file for append");
        }
        tomb_out.write(reinterpret_cast<const char*>(&id), static_cast<std::streamsize>(sizeof(id)));
        if (!tomb_out.good()) {
            return Status::Error("failed appending tombstone");
        }
        it->second.deleted = true;
        record_dirty(it->second.row, reason);
        return Status::Ok();
    }

    Status apply_update_metadata_internal(std::uint64_t id, const std::string& patch_json, const std::string& reason) {
        auto it = entries.find(id);
        if (it == entries.end()) {
            return Status::Error("id not found");
        }
        it->second.metadata_json = patch_json;
        std::ofstream meta_out(seg_meta(active_segment_id), std::ios::binary | std::ios::app);
        if (!meta_out) {
            return Status::Error("failed opening metadata file for update");
        }
        meta_out << "{\"id\":" << id << ",\"metadata\":\"" << json_escape(patch_json) << "\"}\n";
        if (!meta_out.good()) {
            return Status::Error("failed appending metadata update");
        }
        record_dirty(it->second.row, reason);
        return Status::Ok();
    }

    Status replay_wal() {
        std::vector<WalRecord> records;
        if (const Status s = load_wal_records(&records); !s.ok) {
            return s;
        }
        std::sort(records.begin(), records.end(), [](const WalRecord& a, const WalRecord& b) {
            return a.lsn < b.lsn;
        });
        replay_mode = true;
        for (const auto& rec : records) {
            if (rec.lsn <= checkpoint_lsn) {
                continue;
            }
            Status op = Status::Ok();
            if (rec.op == "INSERT") {
                op = apply_insert_internal(rec.id, rec.vector_fp32, rec.metadata_json, true, "replay_insert");
            } else if (rec.op == "DELETE") {
                op = apply_remove_internal(rec.id, "replay_delete");
            } else if (rec.op == "UPDATE_META") {
                op = apply_update_metadata_internal(rec.id, rec.metadata_json, "replay_update_metadata");
            }
            if (!op.ok) {
                replay_mode = false;
                return Status::Error("wal replay failed at lsn " + std::to_string(rec.lsn) + ": " + op.message);
            }
            if (rec.lsn > last_lsn) {
                last_lsn = rec.lsn;
            }
        }
        replay_mode = false;
        return Status::Ok();
    }

    WalStats current_wal_stats() const {
        WalStats st{};
        st.checkpoint_lsn = checkpoint_lsn;
        st.last_lsn = last_lsn;
        const fs::path p = wal_path();
        if (!fs::exists(p)) {
            st.wal_entries = 0;
            return st;
        }
        std::ifstream in(p, std::ios::binary);
        std::string line;
        while (std::getline(in, line)) {
            if (!trim_copy(line).empty()) {
                st.wal_entries += 1;
            }
        }
        return st;
    }

    std::optional<std::vector<float>> read_vector_at_row(std::size_t row) const {
        const fs::path vec = seg_vec(active_segment_id);
        std::ifstream in(vec, std::ios::binary);
        if (!in) {
            return std::nullopt;
        }
        const std::uint64_t byte_off = static_cast<std::uint64_t>(row) * static_cast<std::uint64_t>(kVectorDim) * sizeof(float);
        in.seekg(static_cast<std::streamoff>(byte_off), std::ios::beg);
        if (!in.good()) {
            return std::nullopt;
        }
        std::vector<float> out(kVectorDim, 0.0f);
        in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(kVectorDim * sizeof(float)));
        if (!in.good()) {
            return std::nullopt;
        }
        return out;
    }
};

VectorStore::VectorStore(std::string data_dir) : impl_(new Impl(std::move(data_dir))) {}

VectorStore::~VectorStore() {
    (void)close();
    delete impl_;
}

Status VectorStore::init() {
    if (const Status s = impl_->ensure_dirs(); !s.ok) {
        return s;
    }
    if (!fs::exists(impl_->manifest_path())) {
        if (const Status s = impl_->write_manifest(); !s.ok) {
            return s;
        }
    }
    if (!fs::exists(impl_->dirty_ranges_path())) {
        if (const Status s = impl_->write_dirty_ranges(); !s.ok) {
            return s;
        }
    }
    if (!fs::exists(impl_->wal_path())) {
        std::ofstream wal_out(impl_->wal_path(), std::ios::binary | std::ios::app);
        if (!wal_out) {
            return Status::Error("failed creating wal file");
        }
    }
    return Status::Ok();
}

Status VectorStore::open() {
    if (impl_->opened) {
        return Status::Ok();
    }
    if (const Status s = init(); !s.ok) {
        return s;
    }
    if (impl_->open_signature_unchanged()) {
        impl_->opened = true;
        return Status::Ok();
    }
    if (const Status s = impl_->load_manifest(); !s.ok) {
        return s;
    }
    if (const Status s = impl_->load_ids_and_vectors(); !s.ok) {
        return s;
    }
    if (const Status s = impl_->load_metadata(); !s.ok) {
        return s;
    }
    if (const Status s = impl_->load_tombstones(); !s.ok) {
        return s;
    }
    if (const Status s = impl_->load_dirty_ranges(); !s.ok) {
        return s;
    }
    if (const Status s = impl_->load_cluster_manifest(); !s.ok) {
        return s;
    }
    if (const Status s = impl_->replay_wal(); !s.ok) {
        return s;
    }
    if (const Status s = flush(); !s.ok) {
        return s;
    }
    impl_->last_open_signature = impl_->capture_open_signature();
    impl_->opened = true;
    return Status::Ok();
}

Status VectorStore::flush() {
    if (const Status s = impl_->write_manifest(); !s.ok) {
        return s;
    }
    if (const Status s = impl_->write_dirty_ranges(); !s.ok) {
        return s;
    }
    impl_->last_open_signature = impl_->capture_open_signature();
    return Status::Ok();
}

Status VectorStore::checkpoint() {
    if (!impl_->opened) {
        if (const Status s = open(); !s.ok) {
            return s;
        }
    }
    impl_->checkpoint_lsn = impl_->last_lsn;
    if (const Status s = flush(); !s.ok) {
        return s;
    }
    return write_text_atomic(impl_->wal_path(), "");
}

Status VectorStore::build_initial_clusters(std::uint32_t seed) {
    const auto t_build_start = std::chrono::steady_clock::now();
    if (!impl_->opened) {
        if (const Status s = open(); !s.ok) {
            return s;
        }
    }
    InitialClusteringConfig cfg = impl_->make_clustering_config(seed);

    const auto t_live_start = std::chrono::steady_clock::now();
    const auto live_result = impl_->collect_live_vectors(cfg);
    const auto& live = live_result.vectors_by_id;
    const auto t_live_end = std::chrono::steady_clock::now();
    if (live.size() < 8) {
        return Status::Error("need at least 8 live vectors to build initial clusters");
    }
    std::cout << "progress: clustering collected live vectors=" << live.size() << "\n";
    std::vector<std::vector<float>> vectors;
    vectors.reserve(live.size());
    for (const auto& kv : live) {
        vectors.push_back(kv.second);
    }

    cfg.max_sample = std::max<std::size_t>(256, std::min<std::size_t>(4096, vectors.size()));

    IdEstimateRange idr;
    const auto t_id_start = std::chrono::steady_clock::now();
    std::cout << "progress: clustering step 1/4 id estimation\n";
    if (const Status s = estimate_intrinsic_dimensionality(vectors, cfg.seed, cfg.min_sample, cfg.max_sample, &idr); !s.ok) {
        return s;
    }
    const auto t_id_end = std::chrono::steady_clock::now();
    KMeansModel model;
    ElbowSelection elbow;
    const auto t_elbow_start = std::chrono::steady_clock::now();
    const std::size_t elbow_candidate_count =
        (idr.k_max >= idr.k_min) ? ((idr.k_max - idr.k_min) + 1) : 0;
    std::cout << "progress: elbow integer candidate span "
              << "k_min=" << idr.k_min
              << " k_max=" << idr.k_max
              << " count=" << elbow_candidate_count << "\n";
    std::cout << "progress: clustering step 2/4 binary elbow k selection\n";
    if (const Status s = select_k_binary_elbow_packed(
            vectors,
            live_result.packed_row_major,
            idr,
            cfg,
            &model,
            &elbow);
        !s.ok) {
        return s;
    }
    const auto t_elbow_end = std::chrono::steady_clock::now();
    StabilityMetrics stability;
    const auto t_stability_start = std::chrono::steady_clock::now();
    std::cout << "progress: clustering step 3/4 stability evaluation\n";
    if (const Status s = evaluate_stability_packed(
            vectors,
            live_result.packed_row_major,
            model.k,
            cfg,
            &stability);
        !s.ok) {
        return s;
    }
    const auto t_stability_end = std::chrono::steady_clock::now();

    const std::uint64_t version = impl_->next_cluster_version();
    const auto t_write_start = std::chrono::steady_clock::now();
    std::cout << "progress: clustering step 4/4 writing artifacts version=" << version << "\n";
    if (const Status s = impl_->write_initial_cluster_artifacts(version, idr, elbow, model, stability, live); !s.ok) {
        return s;
    }
    const auto t_write_end = std::chrono::steady_clock::now();

    ClusterStats st{};
    st.available = true;
    st.version = version;
    st.build_lsn = impl_->last_lsn;
    st.vectors_indexed = vectors.size();
    st.chosen_k = model.k;
    st.k_min = idr.k_min;
    st.k_max = idr.k_max;
    st.objective = model.objective;
    st.used_cuda = model.used_cuda;
    st.tensor_core_enabled = model.tensor_core_enabled;
    st.gpu_backend = model.gpu_backend;
    st.scoring_ms_total = model.scoring_ms_total;
    st.scoring_calls = model.scoring_calls;
    st.elbow_k_evaluated_count = elbow.k_evaluated_count;
    st.elbow_stage_a_candidates = elbow.stage_a_candidates;
    st.elbow_stage_b_candidates = elbow.stage_b_candidates;
    st.elbow_early_stop_reason = elbow.early_stop_reason;
    st.stability_runs_executed = stability.runs_executed;
    st.load_live_vectors_ms = std::chrono::duration<double, std::milli>(t_live_end - t_live_start).count();
    st.id_estimation_ms = std::chrono::duration<double, std::milli>(t_id_end - t_id_start).count();
    st.elbow_ms = std::chrono::duration<double, std::milli>(t_elbow_end - t_elbow_start).count();
    st.stability_ms = std::chrono::duration<double, std::milli>(t_stability_end - t_stability_start).count();
    st.write_artifacts_ms = std::chrono::duration<double, std::milli>(t_write_end - t_write_start).count();
    st.total_build_ms = std::chrono::duration<double, std::milli>(t_write_end - t_build_start).count();
    st.live_vector_bytes_read = live_result.bytes_read;
    st.live_vector_contiguous_spans = live_result.contiguous_spans;
    st.live_vector_sparse_reads = live_result.sparse_reads;
    st.live_vector_sparse_fallback = live_result.sparse_fallback_used;
    st.live_vector_async_double_buffer = live_result.async_double_buffer_used;
    st.elbow_stage_a_approx_enabled = elbow.stage_a_approx_enabled;
    st.elbow_stage_a_approx_dim = elbow.stage_a_approx_dim;
    st.elbow_stage_a_approx_stride = elbow.stage_a_approx_stride;
    st.elbow_stage_b_pruned_candidates = elbow.stage_b_pruned_candidates;
    st.elbow_stage_b_window_k_min = elbow.stage_b_window_k_min;
    st.elbow_stage_b_window_k_max = elbow.stage_b_window_k_max;
    st.elbow_stage_b_prune_reason = elbow.stage_b_prune_reason;
    st.elbow_int8_search_enabled = elbow.int8_search_enabled;
    st.elbow_int8_tensor_core_used = elbow.int8_tensor_core_used;
    st.elbow_int8_eval_count = elbow.int8_eval_count;
    st.elbow_int8_scale_mode = cfg.elbow_int8_scale_mode;
    st.elbow_scoring_precision =
        (model.scoring_precision == CudaScorePrecision::INT8) ? "int8-search/fp16-final" : "fp16";

    ClusterHealth health{};
    health.available = true;
    health.passed = stability.passed;
    health.mean_nmi = stability.mean_nmi;
    health.std_nmi = stability.std_nmi;
    health.mean_jaccard = stability.mean_jaccard;
    health.mean_centroid_drift = stability.mean_centroid_drift;
    health.status = stability.passed ? "ok" : "failed";

    if (const Status s = impl_->write_cluster_manifest(st, health, idr, elbow.used_fallback); !s.ok) {
        return s;
    }
    impl_->last_open_signature = impl_->capture_open_signature();
    impl_->cluster_stats_cache = st;
    impl_->cluster_health_cache = health;
    std::cout << "progress: clustering done chosen_k=" << st.chosen_k
              << " backend=" << st.gpu_backend
              << " tensor_core=" << (st.tensor_core_enabled ? "on" : "off")
              << " stability=" << (health.passed ? "pass" : "fail") << "\n";
    return Status::Ok();
}

Status VectorStore::build_second_level_clusters(
    std::uint32_t seed,
    std::optional<std::uint64_t> source_version) {
    const auto t_start = std::chrono::steady_clock::now();
    if (!impl_->opened) {
        if (const Status s = open(); !s.ok) {
            return s;
        }
    }
    const std::uint64_t parent_version = source_version.has_value()
        ? *source_version
        : impl_->cluster_stats_cache.version;
    if (parent_version == 0) {
        return Status::Error("second-level clustering requires an existing first-layer cluster version");
    }

    std::unordered_map<std::uint32_t, std::vector<std::uint64_t>> grouped_ids;
    if (const Status s = impl_->load_top1_assignment_groups(parent_version, &grouped_ids); !s.ok) {
        return s;
    }
    std::vector<std::pair<std::uint32_t, std::vector<std::uint64_t>>> ordered_groups;
    ordered_groups.reserve(grouped_ids.size());
    for (auto& kv : grouped_ids) {
        ordered_groups.push_back({kv.first, std::move(kv.second)});
    }
    std::sort(ordered_groups.begin(), ordered_groups.end(), [](const auto& a, const auto& b) {
        if (a.second.size() != b.second.size()) {
            return a.second.size() > b.second.size();
        }
        return a.first < b.first;
    });

    const std::size_t min_vectors =
        std::max<std::size_t>(8, env_size_value("VECTOR_DB_SECOND_LEVEL_MIN_VECTORS", 8));
    InitialClusteringConfig cfg_base = impl_->make_clustering_config(seed);
    std::vector<Impl::SecondLevelCentroidSummary> summaries;
    summaries.reserve(ordered_groups.size());

    std::cout << "progress: second-level clustering parent_version=" << parent_version
              << " centroids=" << ordered_groups.size() << "\n";
    for (std::size_t i = 0; i < ordered_groups.size(); ++i) {
        const std::uint32_t centroid_id = ordered_groups[i].first;
        const auto& ids = ordered_groups[i].second;
        Impl::SecondLevelCentroidSummary summary{};
        summary.centroid_id = centroid_id;
        summary.source_vectors = ids.size();
        summary.output_dir = impl_->second_level_root(parent_version)
            / ("centroid_" + std::to_string(centroid_id));

        std::cout << "progress: second-level centroid " << centroid_id
                  << " (" << (i + 1) << "/" << ordered_groups.size() << ")"
                  << " source_vectors=" << ids.size() << "\n";

        if (ids.size() < min_vectors) {
            summary.processed = false;
            summary.skipped_reason = "below_min_vectors_threshold";
            summaries.push_back(std::move(summary));
            continue;
        }

        Impl::LiveVectorLoadResult subset;
        if (const Status s = impl_->collect_vectors_for_ids(ids, &subset); !s.ok) {
            return s;
        }
        if (subset.vectors_by_id.size() < min_vectors) {
            summary.processed = false;
            summary.skipped_reason = "insufficient_live_vectors_after_filtering";
            summaries.push_back(std::move(summary));
            continue;
        }
        std::vector<std::vector<float>> vectors;
        vectors.reserve(subset.vectors_by_id.size());
        for (const auto& kv : subset.vectors_by_id) {
            vectors.push_back(kv.second);
        }

        InitialClusteringConfig cfg = cfg_base;
        cfg.seed = seed + centroid_id;
        cfg.max_sample = std::max<std::size_t>(256, std::min<std::size_t>(4096, vectors.size()));

        IdEstimateRange idr;
        const auto t_id_start = std::chrono::steady_clock::now();
        if (const Status s = estimate_intrinsic_dimensionality(
                vectors, cfg.seed, cfg.min_sample, cfg.max_sample, &idr);
            !s.ok) {
            return Status::Error(
                "second-level id estimation failed for centroid "
                + std::to_string(centroid_id) + ": " + s.message);
        }
        // Keep elbow k-grid valid for this centroid subset.
        const std::size_t n_vectors = vectors.size();
        if (n_vectors < 2) {
            summary.processed = false;
            summary.skipped_reason = "too_few_vectors_for_elbow";
            summaries.push_back(std::move(summary));
            continue;
        }
        idr.k_max = std::max<std::size_t>(2, std::min<std::size_t>(idr.k_max, n_vectors));
        idr.k_min = std::max<std::size_t>(1, std::min<std::size_t>(idr.k_min, idr.k_max - 1));
        const auto t_id_end = std::chrono::steady_clock::now();

        KMeansModel model;
        ElbowSelection elbow;
        const auto t_elbow_start = std::chrono::steady_clock::now();
        if (const Status s = select_k_binary_elbow_packed(
                vectors,
                subset.packed_row_major,
                idr,
                cfg,
                &model,
                &elbow);
            !s.ok) {
            return Status::Error(
                "second-level elbow selection failed for centroid "
                + std::to_string(centroid_id) + ": " + s.message);
        }
        const auto t_elbow_end = std::chrono::steady_clock::now();

        StabilityMetrics stability;
        const auto t_stability_start = std::chrono::steady_clock::now();
        if (const Status s = evaluate_stability_packed(
                vectors,
                subset.packed_row_major,
                model.k,
                cfg,
                &stability);
            !s.ok) {
            return Status::Error(
                "second-level stability evaluation failed for centroid "
                + std::to_string(centroid_id) + ": " + s.message);
        }
        const auto t_stability_end = std::chrono::steady_clock::now();

        const auto t_write_start = std::chrono::steady_clock::now();
        if (const Status s = impl_->write_cluster_artifacts_to_dir(
                summary.output_dir,
                idr,
                elbow,
                model,
                stability,
                subset.vectors_by_id);
            !s.ok) {
            return Status::Error(
                "second-level artifact write failed for centroid "
                + std::to_string(centroid_id) + ": " + s.message);
        }
        const auto t_write_end = std::chrono::steady_clock::now();

        ClusterStats st{};
        st.available = true;
        st.version = parent_version;
        st.build_lsn = impl_->last_lsn;
        st.vectors_indexed = vectors.size();
        st.chosen_k = model.k;
        st.k_min = idr.k_min;
        st.k_max = idr.k_max;
        st.objective = model.objective;
        st.used_cuda = model.used_cuda;
        st.tensor_core_enabled = model.tensor_core_enabled;
        st.gpu_backend = model.gpu_backend;
        st.scoring_ms_total = model.scoring_ms_total;
        st.scoring_calls = model.scoring_calls;
        st.elbow_k_evaluated_count = elbow.k_evaluated_count;
        st.elbow_stage_a_candidates = elbow.stage_a_candidates;
        st.elbow_stage_b_candidates = elbow.stage_b_candidates;
        st.elbow_early_stop_reason = elbow.early_stop_reason;
        st.stability_runs_executed = stability.runs_executed;
        st.load_live_vectors_ms = 0.0;
        st.id_estimation_ms = std::chrono::duration<double, std::milli>(t_id_end - t_id_start).count();
        st.elbow_ms = std::chrono::duration<double, std::milli>(t_elbow_end - t_elbow_start).count();
        st.stability_ms = std::chrono::duration<double, std::milli>(t_stability_end - t_stability_start).count();
        st.write_artifacts_ms = std::chrono::duration<double, std::milli>(t_write_end - t_write_start).count();
        st.total_build_ms = st.id_estimation_ms + st.elbow_ms + st.stability_ms + st.write_artifacts_ms;
        st.live_vector_bytes_read = subset.bytes_read;
        st.live_vector_contiguous_spans = subset.contiguous_spans;
        st.live_vector_sparse_reads = subset.sparse_reads;
        st.live_vector_sparse_fallback = subset.sparse_fallback_used;
        st.live_vector_async_double_buffer = subset.async_double_buffer_used;
        st.elbow_stage_a_approx_enabled = elbow.stage_a_approx_enabled;
        st.elbow_stage_a_approx_dim = elbow.stage_a_approx_dim;
        st.elbow_stage_a_approx_stride = elbow.stage_a_approx_stride;
        st.elbow_stage_b_pruned_candidates = elbow.stage_b_pruned_candidates;
        st.elbow_stage_b_window_k_min = elbow.stage_b_window_k_min;
        st.elbow_stage_b_window_k_max = elbow.stage_b_window_k_max;
        st.elbow_stage_b_prune_reason = elbow.stage_b_prune_reason;
        st.elbow_int8_search_enabled = elbow.int8_search_enabled;
        st.elbow_int8_tensor_core_used = elbow.int8_tensor_core_used;
        st.elbow_int8_eval_count = elbow.int8_eval_count;
        st.elbow_int8_scale_mode = cfg.elbow_int8_scale_mode;
        st.elbow_scoring_precision =
            (model.scoring_precision == CudaScorePrecision::INT8) ? "int8-search/fp16-final" : "fp16";

        ClusterHealth health{};
        health.available = true;
        health.passed = stability.passed;
        health.mean_nmi = stability.mean_nmi;
        health.std_nmi = stability.std_nmi;
        health.mean_jaccard = stability.mean_jaccard;
        health.mean_centroid_drift = stability.mean_centroid_drift;
        health.status = stability.passed ? "ok" : "failed";

        if (const Status s = impl_->write_second_level_centroid_manifest(
                summary.output_dir,
                parent_version,
                centroid_id,
                ids.size(),
                st,
                health);
            !s.ok) {
            return s;
        }

        summary.processed = true;
        summary.vectors_indexed = st.vectors_indexed;
        summary.stats = st;
        summary.health = health;
        summaries.push_back(std::move(summary));
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    if (const Status s = impl_->write_second_level_summary(
            parent_version,
            seed,
            summaries,
            total_ms);
        !s.ok) {
        return s;
    }
    std::cout << "progress: second-level clustering done parent_version=" << parent_version
              << " total_centroids=" << summaries.size() << "\n";
    impl_->last_open_signature = impl_->capture_open_signature();
    return Status::Ok();
}

Status VectorStore::close() {
    if (!impl_->opened) {
        return Status::Ok();
    }
    if (const Status s = flush(); !s.ok) {
        return s;
    }
    impl_->opened = false;
    return Status::Ok();
}

Status VectorStore::insert(std::uint64_t id, const std::vector<float>& vector_fp32_1024, const std::string& metadata_json, bool upsert) {
    if (!impl_->opened) {
        if (const Status s = open(); !s.ok) {
            return s;
        }
    }
    if (vector_fp32_1024.size() != kVectorDim) {
        return Status::Error("vector dimension must be exactly 1024");
    }
    if (!looks_like_json(metadata_json)) {
        return Status::Error("metadata_json must be a JSON object/array string");
    }
    const auto existing = impl_->entries.find(id);
    if (existing != impl_->entries.end() && !upsert && !impl_->replay_mode) {
        return Status::Error("duplicate id rejected");
    }
    if (!impl_->replay_mode) {
        if (const Status s = impl_->append_wal_insert(id, vector_fp32_1024, metadata_json); !s.ok) {
            return s;
        }
    }
    const Status applied = impl_->apply_insert_internal(
        id,
        vector_fp32_1024,
        metadata_json,
        upsert || impl_->replay_mode,
        impl_->replay_mode ? "replay_insert" : (upsert ? "upsert" : "insert"));
    if (!applied.ok) {
        return applied;
    }
    return flush();
}

Status VectorStore::insert_batch(const std::vector<Record>& records) {
    if (records.empty()) {
        return Status::Ok();
    }
    if (!impl_->opened) {
        if (const Status s = open(); !s.ok) {
            return s;
        }
    }
    std::unordered_set<std::uint64_t> seen_ids;
    seen_ids.reserve(records.size());
    for (const auto& rec : records) {
        if (rec.vector_fp32.size() != kVectorDim) {
            return Status::Error("vector dimension must be exactly 1024");
        }
        if (!looks_like_json(rec.metadata_json)) {
            return Status::Error("metadata_json must be a JSON object/array string");
        }
        if (!seen_ids.insert(rec.id).second) {
            return Status::Error("duplicate id detected within insert_batch payload");
        }
        if (impl_->entries.find(rec.id) != impl_->entries.end() && !impl_->replay_mode) {
            return Status::Error("duplicate id rejected");
        }
    }
    if (!impl_->replay_mode) {
        if (const Status s = impl_->append_wal_insert_batch(records); !s.ok) {
            return s;
        }
    }
    for (const auto& rec : records) {
        const Status applied = impl_->apply_insert_internal(
            rec.id,
            rec.vector_fp32,
            rec.metadata_json,
            false,
            impl_->replay_mode ? "replay_insert" : "insert_batch");
        if (!applied.ok) {
            return applied;
        }
    }
    return flush();
}

Status VectorStore::remove(std::uint64_t id) {
    if (!impl_->opened) {
        if (const Status s = open(); !s.ok) {
            return s;
        }
    }
    const auto it = impl_->entries.find(id);
    if (it == impl_->entries.end()) {
        return Status::Error("id not found");
    }
    if (it->second.deleted) {
        return Status::Ok();
    }
    if (!impl_->replay_mode) {
        if (const Status s = impl_->append_wal_delete(id); !s.ok) {
            return s;
        }
    }
    if (const Status s = impl_->apply_remove_internal(id, impl_->replay_mode ? "replay_delete" : "delete"); !s.ok) {
        return s;
    }
    return flush();
}

Status VectorStore::update_metadata(std::uint64_t id, const std::string& patch_json) {
    if (!impl_->opened) {
        if (const Status s = open(); !s.ok) {
            return s;
        }
    }
    if (!looks_like_json(patch_json)) {
        return Status::Error("patch_json must be a JSON object/array string");
    }
    if (impl_->entries.find(id) == impl_->entries.end()) {
        return Status::Error("id not found");
    }
    if (!impl_->replay_mode) {
        if (const Status s = impl_->append_wal_update_meta(id, patch_json); !s.ok) {
            return s;
        }
    }
    if (const Status s = impl_->apply_update_metadata_internal(id, patch_json, impl_->replay_mode ? "replay_update_metadata" : "update_metadata"); !s.ok) {
        return s;
    }
    return flush();
}

std::optional<StoredRecord> VectorStore::get(std::uint64_t id) const {
    auto it = impl_->entries.find(id);
    if (it == impl_->entries.end()) {
        return std::nullopt;
    }
    const auto vec = impl_->read_vector_at_row(it->second.row);
    if (!vec.has_value()) {
        return std::nullopt;
    }
    return StoredRecord{id, it->second.deleted, *vec, it->second.metadata_json};
}

Stats VectorStore::stats() const { return impl_->current_stats(); }
WalStats VectorStore::wal_stats() const { return impl_->current_wal_stats(); }
ClusterStats VectorStore::cluster_stats() const { return impl_->cluster_stats_cache; }
ClusterHealth VectorStore::cluster_health() const { return impl_->cluster_health_cache; }

}  // namespace vector_db

