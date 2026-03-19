#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
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

#include "vector_db_v3/ingest_pipeline.hpp"
#include "vector_db_v3/vector_store.hpp"
#include "vector_db_v3/codec/endian.hpp"

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
              << "  bulk-insert-bin --path ... --input <bin> [--batch-size <u32>]\n"
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
              << "  run-full-pipeline --path ... --input <jsonl_or_bin> --input-format <jsonl|bin> [--batch-size <u32>] [--seed <u32>]\n"
              << "  cluster-stats --path ...\n"
              << "  cluster-health --path ...\n";
}

constexpr std::uint32_t kBulkInsertBinMagic = 0x49423356U;  // V3BI (little-endian bytes: 56 33 42 49)
constexpr std::uint16_t kBulkInsertBinVersion = 1U;
constexpr std::size_t kBulkInsertBinHeaderBytes = 18U;       // u32 + u16 + u32 + u64
constexpr std::size_t kBulkInsertBinRecordBytes = 8U + vector_db_v3::kVectorDim * sizeof(float);

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

bool env_truthy(const char* value) {
    if (value == nullptr) {
        return false;
    }
    std::string s(value);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s == "1" || s == "true" || s == "yes" || s == "on";
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

void append_ingest_metrics_if_requested(
    const vector_db_v3::ingest::PipelineStats& stats,
    std::uint32_t batch_size) {
    const char* path = std::getenv("VECTOR_DB_V3_INGEST_METRICS_PATH");
    if (path == nullptr || *path == '\0') {
        return;
    }
    std::ofstream out(path, std::ios::app);
    if (!out) {
        return;
    }
    const char* wal_policy_raw = std::getenv("VECTOR_DB_V3_WAL_COMMIT_POLICY");
    std::string wal_policy = "auto";
    if (wal_policy_raw != nullptr && *wal_policy_raw != '\0') {
        wal_policy = wal_policy_raw;
    }
    out << "{"
        << "\"async_enabled\":" << (stats.async_enabled ? "true" : "false")
        << ",\"pinned_enabled\":" << (stats.pinned_enabled ? "true" : "false")
        << ",\"pinned_mode\":\"" << json_escape(stats.pinned_mode) << "\""
        << ",\"wal_commit_policy\":\"" << json_escape(wal_policy) << "\""
        << ",\"batch_size\":" << batch_size
        << ",\"batches_committed\":" << stats.batches_committed
        << ",\"records_committed\":" << stats.records_committed
        << ",\"peak_queue_depth\":" << stats.peak_queue_depth
        << ",\"producer_wait_ms\":" << std::fixed << std::setprecision(3) << stats.producer_wait_ms
        << ",\"consumer_wait_ms\":" << std::fixed << std::setprecision(3) << stats.consumer_wait_ms
        << ",\"commit_apply_ms\":" << std::fixed << std::setprecision(3) << stats.commit_apply_ms
        << "}\n";
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

class BinaryRecordStreamReader {
public:
    explicit BinaryRecordStreamReader(std::string input_path) : path_(std::move(input_path)) {}

    bool open(std::string* error) {
        in_.open(path_, std::ios::binary);
        if (!in_) {
            if (error != nullptr) {
                *error = "unable to open input binary file";
            }
            return false;
        }
        std::vector<std::uint8_t> header(kBulkInsertBinHeaderBytes, 0U);
        in_.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
        if (!in_ || in_.gcount() != static_cast<std::streamsize>(header.size())) {
            if (error != nullptr) {
                *error = "binary header too short";
            }
            return false;
        }
        const std::uint32_t magic = vector_db_v3::codec::load_le_u32(header.data() + 0U);
        const std::uint16_t version = vector_db_v3::codec::load_le_u16(header.data() + 4U);
        const std::uint32_t record_size = vector_db_v3::codec::load_le_u32(header.data() + 6U);
        remaining_ = vector_db_v3::codec::load_le_u64(header.data() + 10U);
        if (magic != kBulkInsertBinMagic) {
            if (error != nullptr) {
                *error = "binary header magic mismatch";
            }
            return false;
        }
        if (version != kBulkInsertBinVersion) {
            if (error != nullptr) {
                *error = "binary header version mismatch";
            }
            return false;
        }
        if (record_size != static_cast<std::uint32_t>(kBulkInsertBinRecordBytes)) {
            if (error != nullptr) {
                *error = "binary record size mismatch";
            }
            return false;
        }
        std::error_code ec;
        const auto file_size = std::filesystem::file_size(std::filesystem::path(path_), ec);
        if (ec) {
            if (error != nullptr) {
                *error = "unable to inspect binary file size";
            }
            return false;
        }
        if (file_size < kBulkInsertBinHeaderBytes) {
            if (error != nullptr) {
                *error = "binary file size is smaller than header";
            }
            return false;
        }
        const std::uint64_t payload_bytes = static_cast<std::uint64_t>(file_size - kBulkInsertBinHeaderBytes);
        const std::uint64_t expected_bytes = remaining_ * static_cast<std::uint64_t>(kBulkInsertBinRecordBytes);
        if (payload_bytes != expected_bytes) {
            if (error != nullptr) {
                *error = "binary payload size does not match record_count";
            }
            return false;
        }
        row_.assign(kBulkInsertBinRecordBytes, 0U);
        opened_ = true;
        return true;
    }

    bool next(vector_db_v3::Record* out, bool* eof, std::string* error) {
        if (out == nullptr || eof == nullptr) {
            if (error != nullptr) {
                *error = "binary reader invalid output pointers";
            }
            return false;
        }
        if (!opened_) {
            if (error != nullptr) {
                *error = "binary reader not open";
            }
            return false;
        }
        if (remaining_ == 0U) {
            *eof = true;
            return true;
        }
        in_.read(reinterpret_cast<char*>(row_.data()), static_cast<std::streamsize>(row_.size()));
        if (!in_ || in_.gcount() != static_cast<std::streamsize>(row_.size())) {
            if (error != nullptr) {
                *error = "binary record truncated";
            }
            return false;
        }
        out->embedding_id = vector_db_v3::codec::load_le_u64(row_.data());
        out->vector.resize(vector_db_v3::kVectorDim);
        for (std::size_t d = 0; d < vector_db_v3::kVectorDim; ++d) {
            out->vector[d] = vector_db_v3::codec::load_le_f32(row_.data() + 8U + d * sizeof(float));
        }
        --remaining_;
        *eof = remaining_ == 0U;
        if (*eof) {
            char trailing = 0;
            in_.read(&trailing, 1);
            if (in_.gcount() != 0) {
                if (error != nullptr) {
                    *error = "binary file has trailing bytes";
                }
                return false;
            }
        }
        return true;
    }

private:
    std::string path_;
    std::ifstream in_;
    std::uint64_t remaining_ = 0U;
    bool opened_ = false;
    std::vector<std::uint8_t> row_;
};

vector_db_v3::Status run_ingest_pipeline(
    vector_db_v3::VectorStore* store,
    std::uint32_t batch_size,
    const vector_db_v3::ingest::BatchProducer& producer,
    std::size_t* inserted_out,
    std::size_t* batches_out) {
    if (store == nullptr || inserted_out == nullptr || batches_out == nullptr) {
        return vector_db_v3::Status::Error("ingest pipeline: invalid outputs");
    }
    vector_db_v3::ingest::PipelineOptions opts{};
    opts.batch_size = batch_size;
    opts.async_enabled = env_truthy(std::getenv("VECTOR_DB_V3_INGEST_ASYNC_MODE"));
    opts.request_pinned = env_truthy(std::getenv("VECTOR_DB_V3_INGEST_PINNED"));
    vector_db_v3::ingest::PipelineStats stats{};
    const vector_db_v3::Status st = vector_db_v3::ingest::run_pipeline(store, opts, producer, &stats);
    if (!st.ok) {
        return st;
    }
    *inserted_out = stats.records_committed;
    *batches_out = stats.batches_committed;
    append_ingest_metrics_if_requested(stats, batch_size);
    return vector_db_v3::Status::Ok();
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
        std::ifstream in(input, std::ios::binary);
        if (!in) {
            return emit_usage_error("unable to open input jsonl file");
        }
        std::size_t inserted = 0;
        std::size_t batches = 0;
        std::size_t line_no = 0;
        auto producer = [&](std::vector<vector_db_v3::Record>* out_batch, bool* eof) -> vector_db_v3::Status {
            if (out_batch == nullptr || eof == nullptr) {
                return vector_db_v3::Status::Error("jsonl producer invalid outputs");
            }
            out_batch->clear();
            std::string line;
            while (out_batch->size() < batch_size && std::getline(in, line)) {
                ++line_no;
                const std::string t = trim(line);
                if (t.empty()) {
                    continue;
                }
                vector_db_v3::Record rec{};
                std::string parse_error;
                if (!parse_jsonl_record(t, rec, parse_error)) {
                    return vector_db_v3::Status::Error(parse_error + " at line " + std::to_string(line_no), 2);
                }
                out_batch->push_back(std::move(rec));
            }
            *eof = in.eof();
            return vector_db_v3::Status::Ok();
        };
        const auto s = run_ingest_pipeline(&store, batch_size, producer, &inserted, &batches);
        if (!s.ok) {
            return s.code == 2 ? emit_usage_error(s.message) : emit_runtime_error(s.message);
        }
        std::cout << "{\"status\":\"ok\",\"command\":\"bulk-insert\",\"inserted\":" << inserted
                  << ",\"batches\":" << batches << "}\n";
        return 0;
    }

    if (command == "bulk-insert-bin") {
        if (!open_or_fail()) {
            return 1;
        }
        const auto input = get_arg(args, "--input");
        const auto batch_size_arg = get_arg(args, "--batch-size");
        if (input.empty()) {
            return emit_usage_error("bulk-insert-bin requires --input <bin>");
        }
        std::uint32_t batch_size = 1000;
        if (!batch_size_arg.empty()) {
            std::uint32_t tmp = 0;
            if (!parse_u32(batch_size_arg, tmp) || tmp == 0) {
                return emit_usage_error("--batch-size must be >= 1");
            }
            batch_size = tmp;
        }

        std::size_t inserted = 0;
        std::size_t batches = 0;
        BinaryRecordStreamReader reader(input);
        std::string reader_error;
        if (!reader.open(&reader_error)) {
            return emit_usage_error(reader_error);
        }
        auto producer = [&](std::vector<vector_db_v3::Record>* out_batch, bool* eof) -> vector_db_v3::Status {
            if (out_batch == nullptr || eof == nullptr) {
                return vector_db_v3::Status::Error("binary producer invalid outputs");
            }
            out_batch->clear();
            bool reached_eof = false;
            while (out_batch->size() < batch_size && !reached_eof) {
                vector_db_v3::Record rec{};
                std::string parse_error;
                if (!reader.next(&rec, &reached_eof, &parse_error)) {
                    return vector_db_v3::Status::Error(parse_error, 2);
                }
                if (!reached_eof || !rec.vector.empty()) {
                    out_batch->push_back(std::move(rec));
                }
            }
            *eof = reached_eof;
            return vector_db_v3::Status::Ok();
        };
        const auto s = run_ingest_pipeline(&store, batch_size, producer, &inserted, &batches);
        if (!s.ok) {
            return s.code == 2 ? emit_usage_error(s.message) : emit_runtime_error(s.message);
        }

        std::cout << "{\"status\":\"ok\",\"command\":\"bulk-insert-bin\",\"inserted\":" << inserted
                  << ",\"batches\":" << batches << "}\n";
        return 0;
    }

    if (command == "run-full-pipeline") {
        const auto input = get_arg(args, "--input");
        const auto input_format = get_arg(args, "--input-format");
        const auto batch_size_arg = get_arg(args, "--batch-size");
        const auto seed_arg = get_arg(args, "--seed");
        const auto with_search_sanity_arg = get_arg(args, "--with-search-sanity");
        const auto with_cluster_stats_arg = get_arg(args, "--with-cluster-stats");
        const auto query_vec_arg = get_arg(args, "--query-vec");
        if (input.empty()) {
            return emit_usage_error("run-full-pipeline requires --input <jsonl_or_bin>");
        }
        if (input_format != "jsonl" && input_format != "bin") {
            return emit_usage_error("run-full-pipeline requires --input-format <jsonl|bin>");
        }
        std::uint32_t batch_size = 1000;
        if (!batch_size_arg.empty()) {
            std::uint32_t tmp = 0;
            if (!parse_u32(batch_size_arg, tmp) || tmp == 0) {
                return emit_usage_error("--batch-size must be >= 1");
            }
            batch_size = tmp;
        }
        std::uint32_t seed = 1234;
        if (!seed_arg.empty() && !parse_u32(seed_arg, seed)) {
            return emit_usage_error("--seed must be a valid u32");
        }
        const bool with_search_sanity = env_truthy(with_search_sanity_arg.c_str());
        const bool with_cluster_stats = env_truthy(with_cluster_stats_arg.c_str());
        if (with_search_sanity && query_vec_arg.empty()) {
            return emit_usage_error("--with-search-sanity requires --query-vec <file_or_csv>");
        }

        const auto init_status = store.init();
        if (!init_status.ok) {
            return emit_runtime_error(init_status.message);
        }
        if (!open_or_fail()) {
            return 1;
        }

        const auto start = std::chrono::steady_clock::now();
        std::size_t inserted = 0;
        std::size_t batches = 0;
        vector_db_v3::Status ingest_status = vector_db_v3::Status::Ok();

        if (input_format == "jsonl") {
            std::ifstream in(input, std::ios::binary);
            if (!in) {
                return emit_usage_error("unable to open input jsonl file");
            }
            std::size_t line_no = 0;
            auto producer = [&](std::vector<vector_db_v3::Record>* out_batch, bool* eof) -> vector_db_v3::Status {
                if (out_batch == nullptr || eof == nullptr) {
                    return vector_db_v3::Status::Error("jsonl producer invalid outputs");
                }
                out_batch->clear();
                std::string line;
                while (out_batch->size() < batch_size && std::getline(in, line)) {
                    ++line_no;
                    const std::string t = trim(line);
                    if (t.empty()) {
                        continue;
                    }
                    vector_db_v3::Record rec{};
                    std::string parse_error;
                    if (!parse_jsonl_record(t, rec, parse_error)) {
                        return vector_db_v3::Status::Error(parse_error + " at line " + std::to_string(line_no), 2);
                    }
                    out_batch->push_back(std::move(rec));
                }
                *eof = in.eof();
                return vector_db_v3::Status::Ok();
            };
            ingest_status = run_ingest_pipeline(&store, batch_size, producer, &inserted, &batches);
        } else {
            BinaryRecordStreamReader reader(input);
            std::string reader_error;
            if (!reader.open(&reader_error)) {
                return emit_usage_error(reader_error);
            }
            auto producer = [&](std::vector<vector_db_v3::Record>* out_batch, bool* eof) -> vector_db_v3::Status {
                if (out_batch == nullptr || eof == nullptr) {
                    return vector_db_v3::Status::Error("binary producer invalid outputs");
                }
                out_batch->clear();
                bool reached_eof = false;
                while (out_batch->size() < batch_size && !reached_eof) {
                    vector_db_v3::Record rec{};
                    std::string parse_error;
                    if (!reader.next(&rec, &reached_eof, &parse_error)) {
                        return vector_db_v3::Status::Error(parse_error, 2);
                    }
                    if (!reached_eof || !rec.vector.empty()) {
                        out_batch->push_back(std::move(rec));
                    }
                }
                *eof = reached_eof;
                return vector_db_v3::Status::Ok();
            };
            ingest_status = run_ingest_pipeline(&store, batch_size, producer, &inserted, &batches);
        }
        if (!ingest_status.ok) {
            return ingest_status.code == 2 ? emit_usage_error(ingest_status.message) : emit_runtime_error(ingest_status.message);
        }

        vector_db_v3::FullPipelineRunStats pipeline_stats{};
        const auto pipeline_status = store.run_full_pipeline_clustering(seed, &pipeline_stats);
        if (!pipeline_status.ok) {
            return emit_runtime_error(pipeline_status.message);
        }

        std::vector<float> query;
        if (with_search_sanity) {
            std::string parse_error;
            if (!parse_vec_arg_1024(query_vec_arg, query, parse_error)) {
                return emit_usage_error(parse_error);
            }
            const auto sanity = store.search_exact(query, 5U);
            if (sanity.empty()) {
                return emit_runtime_error("search sanity query returned no results");
            }
        }

        const auto elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cout << "{\"status\":\"ok\",\"command\":\"run-full-pipeline\""
                  << ",\"inserted\":" << inserted
                  << ",\"batches\":" << batches
                  << ",\"seed\":" << seed
                  << ",\"stages_planned\":" << pipeline_stats.stages_planned
                  << ",\"stages_executed\":" << pipeline_stats.stages_executed
                  << ",\"stages_completed\":" << pipeline_stats.stages_completed
                  << ",\"failed_stage\":" << (pipeline_stats.failed_stage.empty() ? "null" : ("\"" + json_escape(pipeline_stats.failed_stage) + "\""))
                  << ",\"elapsed_ms_total\":" << std::fixed << std::setprecision(3) << elapsed_ms;
        if (with_search_sanity) {
            std::cout << ",\"search_sanity\":\"ok\"";
        }
        if (with_cluster_stats) {
            const auto st = store.cluster_stats();
            std::cout << ",\"cluster_stats_snapshot\":{\"available\":" << (st.available ? "true" : "false")
                      << ",\"vectors_indexed\":" << st.vectors_indexed
                      << ",\"chosen_k\":" << st.chosen_k << "}";
        }
        std::cout << "}\n";
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
