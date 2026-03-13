#include <algorithm>
#include <cstdint>
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
    if (values.size() != vector_db_v2::kVectorDim) {
        return std::nullopt;
    }
    return values;
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

void print_usage() {
    std::cout << "vectordb_v2 <command> [args]\n"
              << "Commands:\n"
              << "  init --path <data_dir>\n"
              << "  insert --path <data_dir> --id <u64> --vec <file_or_csv>\n"
              << "  bulk-insert --path <data_dir> --input <jsonl>\n"
              << "  delete --path <data_dir> --id <u64>\n"
              << "  get --path <data_dir> --id <u64>\n"
              << "  search --path <data_dir> --vec <file_or_csv> [--topk <u32>]\n"
              << "  stats --path <data_dir>\n"
              << "  checkpoint --path <data_dir>\n"
              << "  wal-stats --path <data_dir>\n"
              << "  build-top-clusters --path <data_dir> [--seed <u32>]\n"
              << "  build-mid-layer-clusters --path <data_dir> [--seed <u32>]\n"
              << "  build-lower-layer-clusters --path <data_dir> [--seed <u32>]\n"
              << "  build-final-layer-clusters --path <data_dir> [--seed <u32>]\n"
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
    const std::string path = get_arg(args, "--path", "data_v2");
    vector_db_v2::VectorStore store(path);

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
        std::cout << "ok: initialized " << path << "\n";
        return 0;
    }

    if (command == "insert") {
        if (!open_or_fail()) {
            return 1;
        }
        const std::string id_str = get_arg(args, "--id");
        const std::string vec_arg = get_arg(args, "--vec");
        if (id_str.empty() || vec_arg.empty()) {
            print_usage();
            return 2;
        }
        const auto vec = parse_vector_1024(vec_arg);
        if (!vec.has_value()) {
            std::cerr << "error: could not parse vector\n";
            return 1;
        }
        const auto s = store.insert(static_cast<std::uint64_t>(std::stoull(id_str)), *vec);
        if (!s.ok) {
            std::cerr << "error: " << s.message << "\n";
            return 1;
        }
        std::cout << "ok: inserted " << id_str << "\n";
        return 0;
    }

    if (command == "bulk-insert") {
        if (!open_or_fail()) {
            return 1;
        }
        const std::string input = get_arg(args, "--input");
        if (input.empty()) {
            print_usage();
            return 2;
        }
        std::ifstream in(input, std::ios::binary);
        if (!in) {
            std::cerr << "error: failed opening input\n";
            return 1;
        }
        std::vector<vector_db_v2::Record> batch;
        batch.reserve(256);
        std::string line;
        std::size_t inserted = 0;
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            const auto id = extract_u64(line, "embedding_id");
            const auto vec_csv = extract_string(line, "vec_csv");
            if (!id.has_value() || !vec_csv.has_value()) {
                std::cerr << "error: invalid jsonl row\n";
                return 1;
            }
            const auto vec = parse_vector_1024(*vec_csv);
            if (!vec.has_value()) {
                std::cerr << "error: invalid vector row\n";
                return 1;
            }
            batch.push_back(vector_db_v2::Record{*id, *vec});
            if (batch.size() >= 256) {
                const auto s = store.insert_batch(batch);
                if (!s.ok) {
                    std::cerr << "error: " << s.message << "\n";
                    return 1;
                }
                inserted += batch.size();
                batch.clear();
            }
        }
        if (!batch.empty()) {
            const auto s = store.insert_batch(batch);
            if (!s.ok) {
                std::cerr << "error: " << s.message << "\n";
                return 1;
            }
            inserted += batch.size();
        }
        std::cout << "ok: bulk inserted " << inserted << "\n";
        return 0;
    }

    if (command == "delete") {
        if (!open_or_fail()) {
            return 1;
        }
        const std::string id_str = get_arg(args, "--id");
        if (id_str.empty()) {
            print_usage();
            return 2;
        }
        const auto s = store.remove(static_cast<std::uint64_t>(std::stoull(id_str)));
        if (!s.ok) {
            std::cerr << "error: " << s.message << "\n";
            return 1;
        }
        std::cout << "ok: deleted " << id_str << "\n";
        return 0;
    }

    if (command == "get") {
        if (!open_or_fail()) {
            return 1;
        }
        const std::string id_str = get_arg(args, "--id");
        if (id_str.empty()) {
            print_usage();
            return 2;
        }
        const auto rec = store.get(static_cast<std::uint64_t>(std::stoull(id_str)));
        if (!rec.has_value()) {
            std::cerr << "error: not found\n";
            return 1;
        }
        std::cout << "{\"embedding_id\": " << rec->embedding_id << ", \"vector_first4\": [";
        for (std::size_t i = 0; i < 4 && i < rec->vector.size(); ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << std::fixed << std::setprecision(6) << rec->vector[i];
        }
        std::cout << "]}\n";
        return 0;
    }

    if (command == "search") {
        if (!open_or_fail()) {
            return 1;
        }
        const std::string vec_arg = get_arg(args, "--vec");
        const std::size_t top_k = static_cast<std::size_t>(std::stoul(get_arg(args, "--topk", "10")));
        const auto vec = parse_vector_1024(vec_arg);
        if (!vec.has_value()) {
            std::cerr << "error: invalid query vector\n";
            return 1;
        }
        const auto results = store.search_exact(*vec, top_k);
        std::cout << "[\n";
        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            std::cout << "  {\"embedding_id\": " << r.embedding_id << ", \"score\": " << std::fixed << std::setprecision(8)
                      << r.score << "}";
            if (i + 1 < results.size()) {
                std::cout << ",";
            }
            std::cout << "\n";
        }
        std::cout << "]\n";
        return 0;
    }

    if (command == "stats") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto st = store.stats();
        std::cout << "{\"dimension\": " << st.dimension << ", \"total_rows\": " << st.total_rows
                  << ", \"live_rows\": " << st.live_rows << ", \"tombstone_rows\": " << st.tombstone_rows << "}\n";
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
        std::cout << "ok: checkpoint\n";
        return 0;
    }

    if (command == "wal-stats") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto st = store.wal_stats();
        std::cout << "{\"checkpoint_lsn\": " << st.checkpoint_lsn << ", \"last_lsn\": " << st.last_lsn
                  << ", \"wal_entries\": " << st.wal_entries << "}\n";
        return 0;
    }

    auto run_cluster = [&](const std::string& cmd) -> int {
        if (!open_or_fail()) {
            return 1;
        }
        const auto seed = static_cast<std::uint32_t>(std::stoul(get_arg(args, "--seed", "1234")));
        vector_db_v2::Status s;
        if (cmd == "top") {
            s = store.build_top_clusters(seed);
        } else if (cmd == "mid") {
            s = store.build_mid_layer_clusters(seed);
        } else if (cmd == "lower") {
            s = store.build_lower_layer_clusters(seed);
        } else {
            s = store.build_final_layer_clusters(seed);
        }
        if (!s.ok) {
            std::cerr << "error: " << s.message << "\n";
            return 1;
        }
        std::cout << "ok: build " << cmd << "\n";
        return 0;
    };

    if (command == "build-top-clusters") {
        return run_cluster("top");
    }
    if (command == "build-mid-layer-clusters") {
        return run_cluster("mid");
    }
    if (command == "build-lower-layer-clusters") {
        return run_cluster("lower");
    }
    if (command == "build-final-layer-clusters") {
        return run_cluster("final");
    }

    if (command == "cluster-stats") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto st = store.cluster_stats();
        std::cout << "{\n"
                  << "  \"available\": " << (st.available ? "true" : "false") << ",\n"
                  << "  \"build_lsn\": " << st.build_lsn << ",\n"
                  << "  \"vectors_indexed\": " << st.vectors_indexed << ",\n"
                  << "  \"chosen_k\": " << st.chosen_k << ",\n"
                  << "  \"k_min\": " << st.k_min << ",\n"
                  << "  \"k_max\": " << st.k_max << ",\n"
                  << "  \"objective\": " << st.objective << ",\n"
                  << "  \"cuda_required\": " << (st.cuda_required ? "true" : "false") << ",\n"
                  << "  \"cuda_enabled\": " << (st.cuda_enabled ? "true" : "false") << ",\n"
                  << "  \"tensor_core_required\": " << (st.tensor_core_required ? "true" : "false") << ",\n"
                  << "  \"tensor_core_active\": " << (st.tensor_core_active ? "true" : "false") << ",\n"
                  << "  \"gpu_arch_class\": \"" << st.gpu_arch_class << "\",\n"
                  << "  \"kernel_backend_path\": \"" << st.kernel_backend_path << "\",\n"
                  << "  \"hot_path_language\": \"" << st.hot_path_language << "\",\n"
                  << "  \"compliance_status\": \"" << st.compliance_status << "\",\n"
                  << "  \"fallback_reason\": \"" << st.fallback_reason << "\",\n"
                  << "  \"non_compliance_stage\": \"" << st.non_compliance_stage << "\"\n"
                  << "}\n";
        return 0;
    }

    if (command == "cluster-health") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto h = store.cluster_health();
        std::cout << "{\"available\": " << (h.available ? "true" : "false")
                  << ", \"passed\": " << (h.passed ? "true" : "false") << ", \"status\": \"" << h.status
                  << "\", \"mean_nmi\": " << h.mean_nmi << ", \"std_nmi\": " << h.std_nmi
                  << ", \"mean_jaccard\": " << h.mean_jaccard
                  << ", \"mean_centroid_drift\": " << h.mean_centroid_drift << "}\n";
        return 0;
    }

    print_usage();
    return 2;
}
