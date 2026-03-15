#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "vector_db_v3/vector_store.hpp"

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

void print_usage() {
    std::cout << "vectordb_v3 <command> [args]\n"
              << "Commands:\n"
              << "  init --path <data_dir>\n"
              << "  insert --path ... --id <u64> --vec <file_or_csv>\n"
              << "  bulk-insert --path ... --input <jsonl> [--batch-size <u32>]\n"
              << "  delete --path ... --id <u64>\n"
              << "  get --path ... --id <u64>\n"
              << "  search --path ... --vec <file_or_csv> [--topk <u32>]\n"
              << "  stats --path ...\n"
              << "  wal-stats --path ...\n"
              << "  checkpoint --path ...\n"
              << "  build-top-clusters --path ... [--seed <u32>]\n"
              << "  build-mid-layer-clusters --path ... [--seed <u32>]\n"
              << "  build-lower-layer-clusters --path ... [--seed <u32>]\n"
              << "  build-final-layer-clusters --path ... [--seed <u32>]\n"
              << "  cluster-stats --path ...\n"
              << "  cluster-health --path ...\n";
}

bool parse_u64(const std::string& s, std::uint64_t& out) {
    try {
        out = static_cast<std::uint64_t>(std::stoull(s));
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_u32(const std::string& s, std::uint32_t& out) {
    try {
        out = static_cast<std::uint32_t>(std::stoul(s));
        return true;
    } catch (...) {
        return false;
    }
}

std::string trim(const std::string& s) {
    const auto left = s.find_first_not_of(" \t\r\n");
    if (left == std::string::npos) {
        return "";
    }
    const auto right = s.find_last_not_of(" \t\r\n");
    return s.substr(left, right - left + 1);
}

std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (const char c : in) {
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

int emit_usage_error(const std::string& message) {
    std::cerr << "error: " << message << "\n";
    return 2;
}

int emit_runtime_error(const std::string& message) {
    std::cerr << "error: " << message << "\n";
    return 1;
}

bool parse_vector_csv_1024(const std::string& csv, std::vector<float>& out, std::string& error) {
    out.clear();
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        const std::string t = trim(token);
        if (t.empty()) {
            error = "vector parse failed: empty token";
            return false;
        }
        try {
            out.push_back(std::stof(t));
        } catch (...) {
            error = "vector parse failed: non-numeric value";
            return false;
        }
    }
    if (out.size() != vector_db_v3::kVectorDim) {
        error = "vector dimension mismatch: expected 1024 values";
        return false;
    }
    return true;
}

bool load_text_file(const std::filesystem::path& p, std::string& out) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        return false;
    }
    std::stringstream buf;
    buf << in.rdbuf();
    out = buf.str();
    return true;
}

bool parse_vec_arg_1024(const std::string& arg, std::vector<float>& out, std::string& error) {
    // Treat comma-containing input as inline CSV immediately.
    if (arg.find(',') != std::string::npos) {
        return parse_vector_csv_1024(arg, out, error);
    }

    std::string source = arg;
    std::error_code ec;
    const bool path_exists = std::filesystem::exists(std::filesystem::path(arg), ec);
    if (ec) {
        error = "vector input path check failed";
        return false;
    }
    if (path_exists) {
        if (!load_text_file(arg, source)) {
            error = "failed to read vector file";
            return false;
        }
    }
    return parse_vector_csv_1024(source, out, error);
}

bool parse_jsonl_record(const std::string& line, vector_db_v3::Record& out, std::string& error) {
    static const std::regex id_re("\"embedding_id\"\\s*:\\s*([0-9]+)");
    static const std::regex vec_re("\"vector\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch m_id;
    std::smatch m_vec;
    if (!std::regex_search(line, m_id, id_re) || m_id.size() < 2) {
        error = "jsonl row missing embedding_id";
        return false;
    }
    if (!std::regex_search(line, m_vec, vec_re) || m_vec.size() < 2) {
        error = "jsonl row missing vector";
        return false;
    }
    try {
        out.embedding_id = static_cast<std::uint64_t>(std::stoull(m_id[1].str()));
    } catch (...) {
        error = "jsonl row embedding_id parse failed";
        return false;
    }
    std::vector<float> vec;
    if (!parse_vector_csv_1024(m_vec[1].str(), vec, error)) {
        return false;
    }
    out.vector = std::move(vec);
    return true;
}

bool parse_jsonl_records(
    const std::string& input_path,
    std::vector<vector_db_v3::Record>& records,
    std::string& error) {
    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        error = "unable to open input jsonl file";
        return false;
    }
    records.clear();
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const std::string t = trim(line);
        if (t.empty()) {
            continue;
        }
        vector_db_v3::Record rec{};
        if (!parse_jsonl_record(t, rec, error)) {
            error += " at line " + std::to_string(line_no);
            return false;
        }
        records.push_back(std::move(rec));
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 2;
    }

    const std::string command = argv[1];
    const Args args = parse_kv_args(argc, argv, 2);
    const std::string path = get_arg(args, "--path", "data_v3");
    vector_db_v3::VectorStore store(path);

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
            return emit_runtime_error(s.message);
        }
        std::cout << "{\"status\":\"ok\",\"command\":\"init\",\"path\":\"" << json_escape(path) << "\"}\n";
        return 0;
    }

    if (command == "insert") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto id_arg = get_arg(args, "--id");
        const auto vec_arg = get_arg(args, "--vec");
        std::uint64_t id = 0;
        if (id_arg.empty() || vec_arg.empty() || !parse_u64(id_arg, id)) {
            return emit_usage_error("insert requires --id <u64> and --vec <file_or_csv>");
        }
        std::vector<float> vec;
        std::string parse_error;
        if (!parse_vec_arg_1024(vec_arg, vec, parse_error)) {
            return emit_usage_error(parse_error);
        }
        const auto s = store.insert(id, vec);
        if (!s.ok) {
            return emit_runtime_error(s.message);
        }
        std::cout << "{\"status\":\"ok\",\"command\":\"insert\",\"embedding_id\":" << id << "}\n";
        return 0;
    }

    if (command == "bulk-insert") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto input = get_arg(args, "--input");
        const auto batch_size_arg = get_arg(args, "--batch-size");
        if (input.empty()) {
            return emit_usage_error("bulk-insert requires --input <jsonl>");
        }
        std::uint32_t batch_size = 1000;
        if (!batch_size_arg.empty()) {
            std::uint32_t tmp = 0;
            if (!parse_u32(batch_size_arg, tmp) || tmp == 0) {
                return emit_usage_error("--batch-size must be >= 1");
            }
            batch_size = tmp;
        }
        std::vector<vector_db_v3::Record> records;
        std::string parse_error;
        if (!parse_jsonl_records(input, records, parse_error)) {
            return emit_usage_error(parse_error);
        }
        std::size_t inserted = 0;
        std::size_t batches = 0;
        std::vector<vector_db_v3::Record> chunk;
        chunk.reserve(batch_size);
        for (const auto& rec : records) {
            chunk.push_back(rec);
            if (chunk.size() >= batch_size) {
                const auto s = store.insert_batch(chunk);
                if (!s.ok) {
                    return emit_runtime_error(s.message);
                }
                inserted += chunk.size();
                ++batches;
                chunk.clear();
            }
        }
        if (!chunk.empty()) {
            const auto s = store.insert_batch(chunk);
            if (!s.ok) {
                return emit_runtime_error(s.message);
            }
            inserted += chunk.size();
            ++batches;
        }
        std::cout << "{\"status\":\"ok\",\"command\":\"bulk-insert\",\"inserted\":" << inserted
                  << ",\"batches\":" << batches << "}\n";
        return 0;
    }

    if (command == "delete") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto id_arg = get_arg(args, "--id");
        std::uint64_t id = 0;
        if (id_arg.empty() || !parse_u64(id_arg, id)) {
            return emit_usage_error("delete requires --id <u64>");
        }
        const auto s = store.remove(id);
        if (!s.ok) {
            return emit_runtime_error(s.message);
        }
        std::cout << "{\"status\":\"ok\",\"command\":\"delete\",\"embedding_id\":" << id << "}\n";
        return 0;
    }

    if (command == "get") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto id_arg = get_arg(args, "--id");
        std::uint64_t id = 0;
        if (id_arg.empty() || !parse_u64(id_arg, id)) {
            return emit_usage_error("get requires --id <u64>");
        }
        const auto rec = store.get(id);
        if (!rec.has_value()) {
            return emit_runtime_error("not found");
        }
        std::cout << "{\"embedding_id\":" << rec->embedding_id << ",\"vector\":[";
        for (std::size_t i = 0; i < rec->vector.size(); ++i) {
            std::cout << std::fixed << std::setprecision(8) << rec->vector[i];
            if (i + 1 < rec->vector.size()) {
                std::cout << ",";
            }
        }
        std::cout << "]}\n";
        return 0;
    }

    if (command == "search") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto vec_arg = get_arg(args, "--vec");
        std::uint32_t topk = 10;
        if (vec_arg.empty()) {
            return emit_usage_error("search requires --vec <file_or_csv>");
        }
        const auto topk_arg = get_arg(args, "--topk");
        if (!topk_arg.empty() && !parse_u32(topk_arg, topk)) {
            return emit_usage_error("search --topk must be a valid u32");
        }
        std::vector<float> query;
        std::string parse_error;
        if (!parse_vec_arg_1024(vec_arg, query, parse_error)) {
            return emit_usage_error(parse_error);
        }
        const auto res = store.search_exact(query, static_cast<std::size_t>(topk));
        std::cout << "[\n";
        for (std::size_t i = 0; i < res.size(); ++i) {
            std::cout << "  {\"embedding_id\": " << res[i].embedding_id << ", \"score\": "
                      << std::fixed << std::setprecision(8) << res[i].score << "}";
            if (i + 1 < res.size()) {
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
        std::cout << "{\"dimension\": " << st.dimension
                  << ", \"total_rows\": " << st.total_rows
                  << ", \"live_rows\": " << st.live_rows
                  << ", \"tombstone_rows\": " << st.tombstone_rows << "}\n";
        return 0;
    }

    if (command == "wal-stats") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto st = store.wal_stats();
        std::cout << "{\"checkpoint_lsn\": " << st.checkpoint_lsn
                  << ", \"last_lsn\": " << st.last_lsn
                  << ", \"wal_entries\": " << st.wal_entries << "}\n";
        return 0;
    }

    if (command == "checkpoint") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto s = store.checkpoint();
        if (!s.ok) {
            return emit_runtime_error(s.message);
        }
        std::cout << "{\"status\":\"ok\",\"command\":\"checkpoint\"}\n";
        return 0;
    }

    auto run_cluster = [&](const std::string& stage) -> int {
        if (!open_or_fail()) {
            return 1;
        }
        std::uint32_t seed = 1234;
        const auto seed_arg = get_arg(args, "--seed");
        if (!seed_arg.empty() && !parse_u32(seed_arg, seed)) {
            return emit_usage_error("cluster build --seed must be a valid u32");
        }
        vector_db_v3::Status s{};
        if (stage == "top") {
            s = store.build_top_clusters(seed);
        } else if (stage == "mid") {
            s = store.build_mid_layer_clusters(seed);
        } else if (stage == "lower") {
            s = store.build_lower_layer_clusters(seed);
        } else {
            s = store.build_final_layer_clusters(seed);
        }
        if (!s.ok) {
            return emit_runtime_error(s.message);
        }
        std::cout << "{\"status\":\"ok\",\"command\":\"build-" << stage << "\",\"seed\":" << seed << "}\n";
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
                  << ", \"passed\": " << (h.passed ? "true" : "false")
                  << ", \"status\": \"" << h.status
                  << "\", \"mean_nmi\": " << h.mean_nmi
                  << ", \"std_nmi\": " << h.std_nmi
                  << ", \"mean_jaccard\": " << h.mean_jaccard
                  << ", \"mean_centroid_drift\": " << h.mean_centroid_drift << "}\n";
        return 0;
    }

    print_usage();
    return 2;
}
