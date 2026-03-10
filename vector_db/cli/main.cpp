#include <cstdlib>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "vector_db/vector_store.hpp"

namespace {

using Args = std::unordered_map<std::string, std::string>;

Args parse_kv_args(int argc, char** argv, int start) {
    Args out;
    for (int i = start; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            continue;
        }
        if (i + 1 < argc) {
            out[key] = argv[i + 1];
            ++i;
        } else {
            out[key] = "";
        }
    }
    return out;
}

std::string get_arg(const Args& args, const std::string& key, const std::string& fallback = "") {
    const auto it = args.find(key);
    if (it == args.end()) {
        return fallback;
    }
    return it->second;
}

std::optional<std::vector<float>> parse_vector_1024(const std::string& file_or_csv) {
    std::string raw = file_or_csv;
    std::ifstream in(file_or_csv, std::ios::binary);
    if (in) {
        std::ostringstream os;
        os << in.rdbuf();
        raw = os.str();
    }

    for (char& c : raw) {
        if (c == '\n' || c == '\r' || c == '\t') {
            c = ',';
        }
    }

    std::vector<float> values;
    std::stringstream ss(raw);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        try {
            values.push_back(std::stof(token));
        } catch (...) {
            return std::nullopt;
        }
    }
    if (values.size() != vector_db::kVectorDim) {
        return std::nullopt;
    }
    return values;
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

std::optional<std::string> extract_json_string_field(const std::string& line, const std::string& key) {
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
    bool esc = false;
    std::size_t q2 = std::string::npos;
    for (std::size_t i = q1 + 1; i < line.size(); ++i) {
        const char c = line[i];
        if (esc) {
            esc = false;
            continue;
        }
        if (c == '\\') {
            esc = true;
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
    return json_unescape(line.substr(q1 + 1, q2 - q1 - 1));
}

std::optional<std::uint64_t> extract_json_u64_field(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = line.find(needle);
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const auto n0 = line.find_first_of("0123456789", key_pos + needle.size());
    if (n0 == std::string::npos) {
        return std::nullopt;
    }
    const auto n1 = line.find_first_not_of("0123456789", n0);
    try {
        return static_cast<std::uint64_t>(std::stoull(line.substr(n0, n1 - n0)));
    } catch (...) {
        return std::nullopt;
    }
}

void print_usage() {
    std::cout << "vectordb <command> [args]\n"
              << "Commands:\n"
              << "  init --path <data_dir>\n"
              << "  insert --path <data_dir> --id <u64> --vec <file_or_csv> --meta <json>\n"
              << "  delete --path <data_dir> --id <u64>\n"
              << "  update-meta --path <data_dir> --id <u64> --meta <json_patch>\n"
              << "  get --path <data_dir> --id <u64>\n"
              << "  stats --path <data_dir>\n"
              << "  checkpoint --path <data_dir>\n"
              << "  wal-stats --path <data_dir>\n"
              << "  bulk-insert --path <data_dir> --input <insert_payloads.jsonl>\n"
              << "  bulk-insert-bin --path <data_dir> --vectors <vectors.fp32bin> --ids <ids.u64bin> --meta <meta.jsonl>\n"
              << "  build-initial-clusters --path <data_dir> [--seed <u32>]\n"
              << "  cluster-stats --path <data_dir>\n"
              << "  cluster-health --path <data_dir>\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 2;
    }

    const std::string command = argv[1];
    const Args args = parse_kv_args(argc, argv, 2);
    const std::string path = get_arg(args, "--path", "data");
    vector_db::VectorStore store(path);

    auto open_or_fail = [&]() -> bool {
        const auto s = store.open();
        if (!s.ok) {
            std::cerr << "error: " << s.message << "\n";
            return false;
        }
        return true;
    };

    if (command == "init") {
        const auto s = store.init();
        if (!s.ok) {
            std::cerr << "error: " << s.message << "\n";
            return 1;
        }
        std::cout << "ok: initialized at " << path << "\n";
        return 0;
    }

    if (command == "insert") {
        const std::string id_str = get_arg(args, "--id");
        const std::string vec_raw = get_arg(args, "--vec");
        const std::string meta = get_arg(args, "--meta", "{}");
        if (id_str.empty() || vec_raw.empty()) {
            print_usage();
            return 2;
        }
        if (!open_or_fail()) {
            return 1;
        }
        const auto vec = parse_vector_1024(vec_raw);
        if (!vec.has_value()) {
            std::cerr << "error: could not parse 1024-d vector input\n";
            return 1;
        }
        const auto s = store.insert(std::stoull(id_str), *vec, meta);
        if (!s.ok) {
            std::cerr << "error: " << s.message << "\n";
            return 1;
        }
        std::cout << "ok: inserted id=" << id_str << "\n";
        return 0;
    }

    if (command == "bulk-insert") {
        const std::string input = get_arg(args, "--input");
        if (input.empty()) {
            print_usage();
            return 2;
        }
        if (!open_or_fail()) {
            return 1;
        }
        std::ifstream in(input, std::ios::binary);
        if (!in) {
            std::cerr << "error: failed opening input file: " << input << "\n";
            return 1;
        }
        std::string line;
        std::size_t line_no = 0;
        std::size_t inserted = 0;
        std::cout << "progress: bulk insert started\n";
        while (std::getline(in, line)) {
            ++line_no;
            if (line.empty()) {
                continue;
            }
            const auto id = extract_json_u64_field(line, "id");
            const auto vec_csv = extract_json_string_field(line, "vec_csv");
            const auto meta_json = extract_json_string_field(line, "meta_json");
            if (!id.has_value() || !vec_csv.has_value() || !meta_json.has_value()) {
                std::cerr << "error: invalid payload at line " << line_no << "\n";
                return 1;
            }
            const auto vec = parse_vector_1024(*vec_csv);
            if (!vec.has_value()) {
                std::cerr << "error: invalid 1024-d vector at line " << line_no << "\n";
                return 1;
            }
            const auto s = store.insert(*id, *vec, *meta_json);
            if (!s.ok) {
                std::cerr << "error: insert failed at line " << line_no << ": " << s.message << "\n";
                return 1;
            }
            ++inserted;
            if (inserted % 500 == 0) {
                std::cout << "progress: inserted " << inserted << " rows\n";
            }
        }
        std::cout << "ok: bulk inserted rows=" << inserted << "\n";
        return 0;
    }

    if (command == "bulk-insert-bin") {
        const std::string vectors_path = get_arg(args, "--vectors");
        const std::string ids_path = get_arg(args, "--ids");
        const std::string meta_path = get_arg(args, "--meta");
        if (vectors_path.empty() || ids_path.empty() || meta_path.empty()) {
            print_usage();
            return 2;
        }
        if (!open_or_fail()) {
            return 1;
        }
        std::ifstream ids_in(ids_path, std::ios::binary);
        std::ifstream vec_in(vectors_path, std::ios::binary);
        std::ifstream meta_in(meta_path, std::ios::binary);
        if (!ids_in || !vec_in || !meta_in) {
            std::cerr << "error: failed opening binary bulk input files\n";
            return 1;
        }
        std::vector<std::uint64_t> ids;
        while (true) {
            std::uint64_t id = 0;
            ids_in.read(reinterpret_cast<char*>(&id), static_cast<std::streamsize>(sizeof(id)));
            if (ids_in.eof()) {
                break;
            }
            if (!ids_in.good()) {
                std::cerr << "error: failed reading ids file\n";
                return 1;
            }
            ids.push_back(id);
        }
        std::vector<std::string> metas;
        std::string mline;
        while (std::getline(meta_in, mline)) {
            if (!mline.empty()) {
                metas.push_back(mline);
            }
        }
        if (ids.empty() || ids.size() != metas.size()) {
            std::cerr << "error: ids/meta row count mismatch\n";
            return 1;
        }
        vec_in.seekg(0, std::ios::end);
        const auto vec_bytes = vec_in.tellg();
        vec_in.seekg(0, std::ios::beg);
        const std::size_t row_bytes = vector_db::kVectorDim * sizeof(float);
        if (vec_bytes <= 0 || static_cast<std::size_t>(vec_bytes) != ids.size() * row_bytes) {
            std::cerr << "error: vectors file size mismatch for 1024-fp32 rows\n";
            return 1;
        }
        std::vector<float> row(vector_db::kVectorDim, 0.0f);
        std::size_t inserted = 0;
        std::cout << "progress: binary bulk insert started\n";
        for (std::size_t i = 0; i < ids.size(); ++i) {
            vec_in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row_bytes));
            if (!vec_in.good()) {
                std::cerr << "error: failed reading vector row " << i << "\n";
                return 1;
            }
            const auto s = store.insert(ids[i], row, metas[i]);
            if (!s.ok) {
                std::cerr << "error: insert failed at row " << i << ": " << s.message << "\n";
                return 1;
            }
            ++inserted;
            if (inserted % 1000 == 0) {
                std::cout << "progress: inserted " << inserted << " rows\n";
            }
        }
        std::cout << "ok: binary bulk inserted rows=" << inserted << "\n";
        return 0;
    }

    if (command == "delete") {
        const std::string id_str = get_arg(args, "--id");
        if (id_str.empty()) {
            print_usage();
            return 2;
        }
        if (!open_or_fail()) {
            return 1;
        }
        const auto s = store.remove(std::stoull(id_str));
        if (!s.ok) {
            std::cerr << "error: " << s.message << "\n";
            return 1;
        }
        std::cout << "ok: deleted id=" << id_str << "\n";
        return 0;
    }

    if (command == "update-meta") {
        const std::string id_str = get_arg(args, "--id");
        const std::string meta = get_arg(args, "--meta");
        if (id_str.empty() || meta.empty()) {
            print_usage();
            return 2;
        }
        if (!open_or_fail()) {
            return 1;
        }
        const auto s = store.update_metadata(std::stoull(id_str), meta);
        if (!s.ok) {
            std::cerr << "error: " << s.message << "\n";
            return 1;
        }
        std::cout << "ok: updated metadata id=" << id_str << "\n";
        return 0;
    }

    if (command == "get") {
        const std::string id_str = get_arg(args, "--id");
        if (id_str.empty()) {
            print_usage();
            return 2;
        }
        if (!open_or_fail()) {
            return 1;
        }
        const auto rec = store.get(std::stoull(id_str));
        if (!rec.has_value()) {
            std::cerr << "error: id not found\n";
            return 1;
        }
        std::cout << "{\n";
        std::cout << "  \"id\": " << rec->id << ",\n";
        std::cout << "  \"deleted\": " << (rec->deleted ? "true" : "false") << ",\n";
        std::cout << "  \"metadata\": " << rec->metadata_json << ",\n";
        std::cout << "  \"vector_first8\": [";
        for (std::size_t i = 0; i < 8 && i < rec->vector_fp32.size(); ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << std::fixed << std::setprecision(6) << rec->vector_fp32[i];
        }
        std::cout << "]\n";
        std::cout << "}\n";
        return 0;
    }

    if (command == "stats") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto st = store.stats();
        std::cout << "{\n"
                  << "  \"dimension\": " << st.dimension << ",\n"
                  << "  \"total_rows\": " << st.total_rows << ",\n"
                  << "  \"live_rows\": " << st.live_rows << ",\n"
                  << "  \"tombstone_rows\": " << st.tombstone_rows << ",\n"
                  << "  \"segments\": " << st.segments << ",\n"
                  << "  \"dirty_ranges\": " << st.dirty_ranges << "\n"
                  << "}\n";
        return 0;
    }

    if (command == "checkpoint") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto s = store.checkpoint();
        if (!s.ok) {
            std::cerr << "error: " << s.message << "\n";
            return 1;
        }
        std::cout << "ok: checkpoint complete\n";
        return 0;
    }

    if (command == "wal-stats") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto st = store.wal_stats();
        std::cout << "{\n"
                  << "  \"checkpoint_lsn\": " << st.checkpoint_lsn << ",\n"
                  << "  \"last_lsn\": " << st.last_lsn << ",\n"
                  << "  \"wal_entries\": " << st.wal_entries << "\n"
                  << "}\n";
        return 0;
    }

    if (command == "build-initial-clusters") {
        if (!open_or_fail()) {
            return 1;
        }
        const std::string seed_str = get_arg(args, "--seed", "1234");
        const auto s = store.build_initial_clusters(static_cast<std::uint32_t>(std::stoul(seed_str)));
        if (!s.ok) {
            std::cerr << "error: " << s.message << "\n";
            return 1;
        }
        std::cout << "ok: initial clusters built\n";
        return 0;
    }

    if (command == "cluster-stats") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto st = store.cluster_stats();
        std::cout << "{\n"
                  << "  \"available\": " << (st.available ? "true" : "false") << ",\n"
                  << "  \"version\": " << st.version << ",\n"
                  << "  \"build_lsn\": " << st.build_lsn << ",\n"
                  << "  \"vectors_indexed\": " << st.vectors_indexed << ",\n"
                  << "  \"chosen_k\": " << st.chosen_k << ",\n"
                  << "  \"k_min\": " << st.k_min << ",\n"
                  << "  \"k_max\": " << st.k_max << ",\n"
                  << "  \"objective\": " << st.objective << ",\n"
                  << "  \"used_cuda\": " << (st.used_cuda ? "true" : "false") << ",\n"
                  << "  \"tensor_core_enabled\": " << (st.tensor_core_enabled ? "true" : "false") << ",\n"
                  << "  \"gpu_backend\": \"" << st.gpu_backend << "\",\n"
                  << "  \"scoring_ms_total\": " << st.scoring_ms_total << ",\n"
                  << "  \"scoring_calls\": " << st.scoring_calls << "\n"
                  << "}\n";
        return 0;
    }

    if (command == "cluster-health") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto h = store.cluster_health();
        std::cout << "{\n"
                  << "  \"available\": " << (h.available ? "true" : "false") << ",\n"
                  << "  \"passed\": " << (h.passed ? "true" : "false") << ",\n"
                  << "  \"mean_nmi\": " << h.mean_nmi << ",\n"
                  << "  \"std_nmi\": " << h.std_nmi << ",\n"
                  << "  \"mean_jaccard\": " << h.mean_jaccard << ",\n"
                  << "  \"mean_centroid_drift\": " << h.mean_centroid_drift << ",\n"
                  << "  \"status\": \"" << h.status << "\"\n"
                  << "}\n";
        return 0;
    }

    print_usage();
    return 2;
}

