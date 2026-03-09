#include "vector_db/vector_store.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
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
    explicit Impl(std::string root) : data_dir(std::move(root)) {}

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

    fs::path root() const { return fs::path(data_dir); }
    fs::path segments_dir() const { return root() / "segments"; }
    fs::path manifest_path() const { return root() / "manifest.json"; }
    fs::path dirty_ranges_path() const { return root() / "dirty_ranges.json"; }
    fs::path wal_path() const { return root() / "wal.log"; }

    fs::path seg_base(std::uint64_t seg_id) const {
        return segments_dir() / ("seg_" + std::to_string(seg_id));
    }

    fs::path seg_vec(std::uint64_t seg_id) const { return seg_base(seg_id).string() + ".vec"; }
    fs::path seg_ids(std::uint64_t seg_id) const { return seg_base(seg_id).string() + ".ids"; }
    fs::path seg_meta(std::uint64_t seg_id) const { return seg_base(seg_id).string() + ".meta.jsonl"; }
    fs::path seg_tomb(std::uint64_t seg_id) const { return seg_base(seg_id).string() + ".tomb"; }

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
    if (const Status s = impl_->replay_wal(); !s.ok) {
        return s;
    }
    if (const Status s = flush(); !s.ok) {
        return s;
    }
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

}  // namespace vector_db

