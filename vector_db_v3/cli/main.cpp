#include <cstdint>
#include <iomanip>
#include <iostream>
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
        const auto id_arg = get_arg(args, "--id");
        const auto vec_arg = get_arg(args, "--vec");
        std::uint64_t id = 0;
        if (id_arg.empty() || vec_arg.empty() || !parse_u64(id_arg, id)) {
            print_usage();
            return 2;
        }
        const auto s = store.insert(id, {});
        if (!s.ok) {
            std::cerr << "error: " << s.message << "\n";
            return 1;
        }
        std::cout << "ok: inserted " << id << "\n";
        return 0;
    }

    if (command == "bulk-insert") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto input = get_arg(args, "--input");
        const auto batch_size_arg = get_arg(args, "--batch-size");
        if (input.empty()) {
            print_usage();
            return 2;
        }
        if (!batch_size_arg.empty()) {
            std::uint32_t tmp = 0;
            if (!parse_u32(batch_size_arg, tmp) || tmp == 0) {
                std::cerr << "error: --batch-size must be >= 1\n";
                return 1;
            }
        }
        const auto s = store.insert_batch({});
        if (!s.ok) {
            std::cerr << "error: " << s.message << "\n";
            return 1;
        }
        std::cout << "ok: bulk inserted 0\n";
        return 0;
    }

    if (command == "delete") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto id_arg = get_arg(args, "--id");
        std::uint64_t id = 0;
        if (id_arg.empty() || !parse_u64(id_arg, id)) {
            print_usage();
            return 2;
        }
        const auto s = store.remove(id);
        if (!s.ok) {
            std::cerr << "error: " << s.message << "\n";
            return 1;
        }
        std::cout << "ok: deleted " << id << "\n";
        return 0;
    }

    if (command == "get") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto id_arg = get_arg(args, "--id");
        std::uint64_t id = 0;
        if (id_arg.empty() || !parse_u64(id_arg, id)) {
            print_usage();
            return 2;
        }
        const auto rec = store.get(id);
        if (!rec.has_value()) {
            std::cerr << "error: not found\n";
            return 1;
        }
        std::cout << "{\"embedding_id\": " << rec->embedding_id << "}\n";
        return 0;
    }

    if (command == "search") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto vec_arg = get_arg(args, "--vec");
        std::uint32_t topk = 10;
        if (vec_arg.empty()) {
            print_usage();
            return 2;
        }
        const auto topk_arg = get_arg(args, "--topk");
        if (!topk_arg.empty() && !parse_u32(topk_arg, topk)) {
            print_usage();
            return 2;
        }
        const auto res = store.search_exact({}, static_cast<std::size_t>(topk));
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
            std::cerr << "error: " << s.message << "\n";
            return 1;
        }
        std::cout << "ok: checkpoint\n";
        return 0;
    }

    auto run_cluster = [&](const std::string& stage) -> int {
        if (!open_or_fail()) {
            return 1;
        }
        std::uint32_t seed = 1234;
        const auto seed_arg = get_arg(args, "--seed");
        if (!seed_arg.empty() && !parse_u32(seed_arg, seed)) {
            print_usage();
            return 2;
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
            std::cerr << "error: " << s.message << "\n";
            return 1;
        }
        std::cout << "ok: build " << stage << "\n";
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
