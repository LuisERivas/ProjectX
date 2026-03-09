#include <cstdlib>
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

void print_usage() {
    std::cout << "vectordb <command> [args]\n"
              << "Commands:\n"
              << "  init --path <data_dir>\n"
              << "  insert --path <data_dir> --id <u64> --vec <file_or_csv> --meta <json>\n"
              << "  delete --path <data_dir> --id <u64>\n"
              << "  update-meta --path <data_dir> --id <u64> --meta <json_patch>\n"
              << "  get --path <data_dir> --id <u64>\n"
              << "  stats --path <data_dir>\n";
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

    print_usage();
    return 2;
}

