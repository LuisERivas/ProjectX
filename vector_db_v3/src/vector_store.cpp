#include "vector_db_v3/vector_store.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iomanip>
#include <map>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vector_db_v3/codec/checksum.hpp"
#include "vector_db_v3/codec/artifacts.hpp"
#include "vector_db_v3/codec/endian.hpp"
#include "vector_db_v3/codec/io.hpp"
#include "vector_db_v3/paths.hpp"
#include "vector_db_v3/telemetry.hpp"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace vector_db_v3 {

namespace {

constexpr std::uint32_t kWalMagic = 0x4C415756U;         // VWAL
constexpr std::uint16_t kWalSchemaVersion = 1U;
constexpr std::uint8_t kWalOpInsert = 1U;
constexpr std::uint8_t kWalOpDelete = 2U;
constexpr std::size_t kWalFixedBytes = 32U;              // through payload_bytes
constexpr std::size_t kWalMinRecordBytes = kWalFixedBytes + 4U;

constexpr std::uint32_t kCheckpointMagic = 0x504B4356U;  // VCKP
constexpr std::uint16_t kCheckpointSchemaVersion = 1U;
constexpr std::size_t kCheckpointHeaderBytes = 36U;

struct ManifestMeta {
    std::uint64_t checkpoint_lsn = 0;
    std::uint64_t last_lsn = 0;
    std::string checkpoint_file;
};

struct WalEntry {
    std::uint64_t lsn = 0;
    std::uint8_t op = 0;
    std::uint64_t embedding_id = 0;
    std::vector<float> vector;
};

bool sync_file_descriptor(const fs::path& p) {
#ifdef _WIN32
    const int fd = _open(p.string().c_str(), _O_RDONLY | _O_BINARY);
    if (fd < 0) {
        return false;
    }
    const int rc = _commit(fd);
    _close(fd);
    return rc == 0;
#else
    const int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
    const int rc = ::fsync(fd);
    ::close(fd);
    return rc == 0;
#endif
}

Status write_manifest_json_atomic(const fs::path& path, const ManifestMeta& meta) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"status\": \"active\",\n"
        << "  \"durability_version\": 1,\n"
        << "  \"checkpoint_lsn\": " << meta.checkpoint_lsn << ",\n"
        << "  \"last_lsn\": " << meta.last_lsn << ",\n"
        << "  \"checkpoint_file\": \"" << meta.checkpoint_file << "\"\n"
        << "}\n";
    const std::string s = out.str();
    const std::vector<std::uint8_t> bytes(s.begin(), s.end());
    return codec::write_atomic_bytes(path, bytes);
}

std::uint64_t parse_u64_or_default(const std::string& body, const std::string& key, std::uint64_t fallback) {
    const std::regex re("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch m;
    if (!std::regex_search(body, m, re) || m.size() < 2) {
        return fallback;
    }
    try {
        return static_cast<std::uint64_t>(std::stoull(m[1].str()));
    } catch (...) {
        return fallback;
    }
}

std::string parse_string_or_default(const std::string& body, const std::string& key, const std::string& fallback) {
    const std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch m;
    if (!std::regex_search(body, m, re) || m.size() < 2) {
        return fallback;
    }
    return m[1].str();
}

Status load_manifest_json(const fs::path& path, ManifestMeta* out) {
    if (out == nullptr) {
        return Status::Error("load_manifest_json: out is null");
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return Status::Error("manifest.json missing; run init first");
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string body = buffer.str();
    out->checkpoint_lsn = parse_u64_or_default(body, "checkpoint_lsn", 0);
    out->last_lsn = parse_u64_or_default(body, "last_lsn", 0);
    out->checkpoint_file = parse_string_or_default(body, "checkpoint_file", "");
    return Status::Ok();
}

Status write_truncated_file(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return Status::Error("failed truncating file " + path.string());
    }
    if (!bytes.empty()) {
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    out.flush();
    if (!out || !sync_file_descriptor(path)) {
        return Status::Error("failed syncing truncated file " + path.string());
    }
    return Status::Ok();
}

Status encode_wal_entry_bytes(const WalEntry& entry, std::vector<std::uint8_t>* out) {
    if (out == nullptr) {
        return Status::Error("encode_wal_entry_bytes: out is null");
    }
    const bool is_insert = entry.op == kWalOpInsert;
    const bool is_delete = entry.op == kWalOpDelete;
    if (!is_insert && !is_delete) {
        return Status::Error("encode_wal_entry_bytes: invalid op");
    }
    if (is_insert && entry.vector.size() != kVectorDim) {
        return Status::Error("encode_wal_entry_bytes: insert vector dim mismatch");
    }
    if (is_delete && !entry.vector.empty()) {
        return Status::Error("encode_wal_entry_bytes: delete should not carry vector payload");
    }

    std::vector<std::uint8_t> payload;
    if (is_insert) {
        payload.resize(kVectorDim * sizeof(float), 0U);
        for (std::size_t i = 0; i < kVectorDim; ++i) {
            codec::store_le_f32(payload.data() + i * sizeof(float), entry.vector[i]);
        }
    }
    const std::uint32_t payload_bytes = static_cast<std::uint32_t>(payload.size());
    const std::uint32_t vector_dim = is_insert ? static_cast<std::uint32_t>(kVectorDim) : 0U;

    out->assign(kWalFixedBytes + payload_bytes + 4U, 0U);
    codec::store_le_u32(out->data() + 0U, kWalMagic);
    codec::store_le_u16(out->data() + 4U, kWalSchemaVersion);
    (*out)[6U] = entry.op;
    (*out)[7U] = 0U;
    codec::store_le_u64(out->data() + 8U, entry.lsn);
    codec::store_le_u64(out->data() + 16U, entry.embedding_id);
    codec::store_le_u32(out->data() + 24U, vector_dim);
    codec::store_le_u32(out->data() + 28U, payload_bytes);
    if (!payload.empty()) {
        std::memcpy(out->data() + kWalFixedBytes, payload.data(), payload.size());
    }
    const std::uint32_t crc = codec::crc32(out->data() + 4U, kWalFixedBytes - 4U + payload.size());
    codec::store_le_u32(out->data() + kWalFixedBytes + payload.size(), crc);
    return Status::Ok();
}

Status append_wal_entry(const fs::path& wal_path, const WalEntry& entry) {
    std::vector<std::uint8_t> bytes;
    const Status enc = encode_wal_entry_bytes(entry, &bytes);
    if (!enc.ok) {
        return enc;
    }
    std::ofstream out(wal_path, std::ios::binary | std::ios::app);
    if (!out) {
        return Status::Error("append_wal_entry: unable to open wal.log");
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    out.flush();
    if (!out) {
        return Status::Error("append_wal_entry: write failed");
    }
    if (!sync_file_descriptor(wal_path)) {
        return Status::Error("append_wal_entry: fsync/_commit failed");
    }
    return Status::Ok();
}

Status load_checkpoint_snapshot(
    const fs::path& checkpoint_path,
    std::uint64_t* checkpoint_lsn,
    std::map<std::uint64_t, Record>* rows) {
    if (checkpoint_lsn == nullptr || rows == nullptr) {
        return Status::Error("load_checkpoint_snapshot: invalid outputs");
    }
    std::vector<std::uint8_t> bytes;
    const Status rd = codec::read_file_bytes(checkpoint_path, &bytes);
    if (!rd.ok) {
        return rd;
    }
    if (bytes.size() < kCheckpointHeaderBytes) {
        return Status::Error("checkpoint corrupted: header too short");
    }
    const std::uint32_t magic = codec::load_le_u32(bytes.data() + 0U);
    const std::uint16_t version = codec::load_le_u16(bytes.data() + 4U);
    const std::uint16_t reserved = codec::load_le_u16(bytes.data() + 6U);
    const std::uint64_t lsn = codec::load_le_u64(bytes.data() + 8U);
    const std::uint64_t row_count = codec::load_le_u64(bytes.data() + 16U);
    const std::uint64_t payload_bytes_u64 = codec::load_le_u64(bytes.data() + 24U);
    const std::uint32_t payload_crc = codec::load_le_u32(bytes.data() + 32U);
    if (magic != kCheckpointMagic || version != kCheckpointSchemaVersion || reserved != 0U) {
        return Status::Error("checkpoint corrupted: bad magic/version/reserved");
    }
    const std::size_t payload_bytes = static_cast<std::size_t>(payload_bytes_u64);
    if (bytes.size() != kCheckpointHeaderBytes + payload_bytes) {
        return Status::Error("checkpoint corrupted: payload size mismatch");
    }
    const std::size_t expected_row_bytes = sizeof(std::uint64_t) + kVectorDim * sizeof(float);
    if (payload_bytes != static_cast<std::size_t>(row_count) * expected_row_bytes) {
        return Status::Error("checkpoint corrupted: row_count mismatch");
    }
    const std::uint32_t crc = codec::crc32(bytes.data() + kCheckpointHeaderBytes, payload_bytes);
    if (crc != payload_crc) {
        return Status::Error("checkpoint corrupted: payload checksum mismatch");
    }

    rows->clear();
    for (std::size_t i = 0; i < static_cast<std::size_t>(row_count); ++i) {
        const std::size_t row_base = kCheckpointHeaderBytes + i * expected_row_bytes;
        const std::uint64_t id = codec::load_le_u64(bytes.data() + row_base);
        std::vector<float> vec(kVectorDim, 0.0f);
        const std::size_t vec_base = row_base + sizeof(std::uint64_t);
        for (std::size_t d = 0; d < kVectorDim; ++d) {
            vec[d] = codec::load_le_f32(bytes.data() + vec_base + d * sizeof(float));
        }
        (*rows)[id] = Record{id, std::move(vec)};
    }
    *checkpoint_lsn = lsn;
    return Status::Ok();
}

Status write_checkpoint_snapshot(
    const fs::path& path,
    std::uint64_t checkpoint_lsn,
    const std::map<std::uint64_t, Record>& rows) {
    const std::size_t row_bytes = sizeof(std::uint64_t) + kVectorDim * sizeof(float);
    std::vector<std::uint8_t> payload(rows.size() * row_bytes, 0U);
    std::size_t idx = 0;
    for (const auto& kv : rows) {
        const std::size_t base = idx * row_bytes;
        codec::store_le_u64(payload.data() + base, kv.first);
        const Record& rec = kv.second;
        if (rec.vector.size() != kVectorDim) {
            return Status::Error("write_checkpoint_snapshot: record has invalid vector dim");
        }
        const std::size_t vec_base = base + sizeof(std::uint64_t);
        for (std::size_t d = 0; d < kVectorDim; ++d) {
            codec::store_le_f32(payload.data() + vec_base + d * sizeof(float), rec.vector[d]);
        }
        ++idx;
    }

    std::vector<std::uint8_t> out(kCheckpointHeaderBytes + payload.size(), 0U);
    codec::store_le_u32(out.data() + 0U, kCheckpointMagic);
    codec::store_le_u16(out.data() + 4U, kCheckpointSchemaVersion);
    codec::store_le_u16(out.data() + 6U, 0U);
    codec::store_le_u64(out.data() + 8U, checkpoint_lsn);
    codec::store_le_u64(out.data() + 16U, static_cast<std::uint64_t>(rows.size()));
    codec::store_le_u64(out.data() + 24U, static_cast<std::uint64_t>(payload.size()));
    codec::store_le_u32(out.data() + 32U, codec::crc32(payload));
    if (!payload.empty()) {
        std::memcpy(out.data() + kCheckpointHeaderBytes, payload.data(), payload.size());
    }
    return codec::write_atomic_bytes(path, out);
}

Status replay_wal_entries(
    const fs::path& wal_path,
    std::uint64_t checkpoint_lsn,
    std::uint64_t* last_lsn,
    std::size_t* wal_entries,
    std::size_t* tombstone_rows,
    std::map<std::uint64_t, Record>* rows) {
    if (last_lsn == nullptr || wal_entries == nullptr || tombstone_rows == nullptr || rows == nullptr) {
        return Status::Error("replay_wal_entries: invalid outputs");
    }
    std::vector<std::uint8_t> bytes;
    const Status rd = codec::read_file_bytes(wal_path, &bytes);
    if (!rd.ok) {
        return rd;
    }
    std::size_t offset = 0;
    std::size_t last_valid_offset = 0;
    std::uint64_t prev_lsn = 0;
    std::size_t parsed_entries = 0;
    std::size_t tombstones = 0;

    while (offset < bytes.size()) {
        const std::size_t remain = bytes.size() - offset;
        if (remain < kWalMinRecordBytes) {
            bytes.resize(last_valid_offset);
            const Status trunc = write_truncated_file(wal_path, bytes);
            if (!trunc.ok) {
                return trunc;
            }
            break;
        }

        const std::uint8_t* p = bytes.data() + offset;
        const std::uint32_t magic = codec::load_le_u32(p + 0U);
        if (magic != kWalMagic) {
            return Status::Error("wal corruption: magic mismatch");
        }
        const std::uint16_t version = codec::load_le_u16(p + 4U);
        const std::uint8_t op = p[6U];
        const std::uint8_t reserved = p[7U];
        const std::uint64_t lsn = codec::load_le_u64(p + 8U);
        const std::uint64_t embedding_id = codec::load_le_u64(p + 16U);
        const std::uint32_t vector_dim = codec::load_le_u32(p + 24U);
        const std::uint32_t payload_bytes = codec::load_le_u32(p + 28U);
        const std::size_t record_bytes = kWalFixedBytes + payload_bytes + 4U;
        if (remain < record_bytes) {
            bytes.resize(last_valid_offset);
            const Status trunc = write_truncated_file(wal_path, bytes);
            if (!trunc.ok) {
                return trunc;
            }
            break;
        }
        if (version != kWalSchemaVersion || reserved != 0U) {
            return Status::Error("wal corruption: bad version/reserved");
        }
        const std::uint32_t stored_crc = codec::load_le_u32(p + kWalFixedBytes + payload_bytes);
        const std::uint32_t calc_crc = codec::crc32(p + 4U, kWalFixedBytes - 4U + payload_bytes);
        if (stored_crc != calc_crc) {
            return Status::Error("wal corruption: checksum mismatch");
        }
        if (prev_lsn != 0 && lsn <= prev_lsn) {
            return Status::Error("wal corruption: non-monotonic LSN sequence");
        }
        prev_lsn = lsn;

        if (op == kWalOpInsert) {
            if (vector_dim != static_cast<std::uint32_t>(kVectorDim) ||
                payload_bytes != static_cast<std::uint32_t>(kVectorDim * sizeof(float))) {
                return Status::Error("wal corruption: invalid insert payload");
            }
            if (lsn > checkpoint_lsn) {
                std::vector<float> vec(kVectorDim, 0.0f);
                for (std::size_t d = 0; d < kVectorDim; ++d) {
                    vec[d] = codec::load_le_f32(p + kWalFixedBytes + d * sizeof(float));
                }
                (*rows)[embedding_id] = Record{embedding_id, std::move(vec)};
            }
        } else if (op == kWalOpDelete) {
            if (vector_dim != 0U || payload_bytes != 0U) {
                return Status::Error("wal corruption: invalid delete payload");
            }
            if (lsn > checkpoint_lsn) {
                rows->erase(embedding_id);
                ++tombstones;
            }
        } else {
            return Status::Error("wal corruption: unknown op code");
        }

        ++parsed_entries;
        *last_lsn = std::max(*last_lsn, lsn);
        last_valid_offset = offset + record_bytes;
        offset = last_valid_offset;
    }

    *wal_entries = parsed_entries;
    *tombstone_rows = tombstones;
    return Status::Ok();
}

struct KMeansResult {
    std::vector<std::vector<float>> centroids;
    std::vector<std::uint32_t> assignments;
    double objective = 0.0;
};

double squared_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double out = 0.0;
    for (std::size_t i = 0; i < kVectorDim; ++i) {
        const double delta = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        out += delta * delta;
    }
    return out;
}

KMeansResult run_deterministic_kmeans(
    const std::vector<std::vector<float>>& vectors,
    std::uint32_t k,
    std::uint32_t max_iterations) {
    KMeansResult out{};
    if (vectors.empty() || k == 0U) {
        return out;
    }
    k = std::min<std::uint32_t>(k, static_cast<std::uint32_t>(vectors.size()));
    out.centroids.assign(k, std::vector<float>(kVectorDim, 0.0f));
    out.assignments.assign(vectors.size(), 0U);

    for (std::uint32_t c = 0; c < k; ++c) {
        const std::size_t idx = (static_cast<std::size_t>(c) * vectors.size()) / k;
        out.centroids[c] = vectors[idx];
    }

    for (std::uint32_t iter = 0; iter < std::max<std::uint32_t>(1U, max_iterations); ++iter) {
        bool changed = false;
        std::vector<double> min_dist(vectors.size(), std::numeric_limits<double>::infinity());
        for (std::size_t i = 0; i < vectors.size(); ++i) {
            std::uint32_t best = 0;
            double best_dist = std::numeric_limits<double>::infinity();
            for (std::uint32_t c = 0; c < k; ++c) {
                const double dist = squared_l2(vectors[i], out.centroids[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            min_dist[i] = best_dist;
            if (iter == 0 || out.assignments[i] != best) {
                changed = true;
                out.assignments[i] = best;
            }
        }

        std::vector<std::vector<double>> sums(k, std::vector<double>(kVectorDim, 0.0));
        std::vector<std::uint32_t> counts(k, 0U);
        for (std::size_t i = 0; i < vectors.size(); ++i) {
            const std::uint32_t bucket = out.assignments[i];
            ++counts[bucket];
            for (std::size_t d = 0; d < kVectorDim; ++d) {
                sums[bucket][d] += static_cast<double>(vectors[i][d]);
            }
        }
        for (std::uint32_t c = 0; c < k; ++c) {
            if (counts[c] == 0U) {
                std::size_t worst_idx = 0;
                double worst_dist = -1.0;
                for (std::size_t i = 0; i < min_dist.size(); ++i) {
                    if (min_dist[i] > worst_dist) {
                        worst_dist = min_dist[i];
                        worst_idx = i;
                    }
                }
                out.assignments[worst_idx] = c;
                counts[c] = 1U;
                for (std::size_t d = 0; d < kVectorDim; ++d) {
                    sums[c][d] = static_cast<double>(vectors[worst_idx][d]);
                }
            }
            for (std::size_t d = 0; d < kVectorDim; ++d) {
                out.centroids[c][d] = static_cast<float>(sums[c][d] / static_cast<double>(counts[c]));
            }
        }
        if (!changed) {
            break;
        }
    }

    out.objective = 0.0;
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        out.objective += squared_l2(vectors[i], out.centroids[out.assignments[i]]);
    }
    return out;
}

std::vector<std::uint32_t> coarse_ks(std::uint32_t k_min, std::uint32_t k_max) {
    std::vector<std::uint32_t> out;
    out.push_back(k_min);
    if (k_max > k_min) {
        out.push_back(k_min + ((k_max - k_min) / 2U));
        out.push_back(k_max);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::string escape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char ch : s) {
        if (ch == '\\' || ch == '"') {
            out.push_back('\\');
        }
        out.push_back(ch);
    }
    return out;
}

std::uint32_t derive_k_min(std::size_t n) {
    if (n == 0U) {
        return 0U;
    }
    return std::max<std::uint32_t>(1U, std::min<std::uint32_t>(32U, static_cast<std::uint32_t>(n)));
}

std::uint32_t derive_k_max(std::size_t n, std::uint32_t k_min) {
    if (n == 0U) {
        return 0U;
    }
    const std::uint32_t upper = std::min<std::uint32_t>(256U, static_cast<std::uint32_t>(n));
    return std::max<std::uint32_t>(k_min, upper);
}

struct StageBranchKey {
    std::uint32_t parent_top = 0;
    std::uint32_t mid_centroid = 0;
    bool operator<(const StageBranchKey& other) const {
        if (parent_top != other.parent_top) {
            return parent_top < other.parent_top;
        }
        return mid_centroid < other.mid_centroid;
    }
};

struct LowerGateOutcome {
    StageBranchKey branch;
    std::uint32_t centroid_id = 0;
    std::string job_id;
    std::uint32_t dataset_size = 0;
    codec::GateDecision gate_decision = codec::GateDecision::NotApplicable;
};

struct MidParentJob {
    std::uint32_t centroid_id = 0;
    std::string job_id;
    std::uint32_t dataset_size = 0;
    std::uint32_t k_min = 0;
    std::uint32_t k_max = 0;
    std::uint32_t chosen_k = 0;
};

Status read_manifest_payload_string(const fs::path& artifact_path, std::string* payload_out) {
    if (payload_out == nullptr) {
        return Status::Error("manifest payload read: invalid output pointer");
    }
    codec::CommonHeader header{};
    std::vector<std::uint8_t> payload;
    const Status st = codec::read_cluster_manifest_file(artifact_path, &header, &payload);
    if (!st.ok) {
        return st;
    }
    *payload_out = std::string(payload.begin(), payload.end());
    return Status::Ok();
}

std::uint32_t parse_u32_or_default(const std::string& body, const std::string& key, std::uint32_t fallback) {
    const std::regex re("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch m;
    if (!std::regex_search(body, m, re) || m.size() < 2) {
        return fallback;
    }
    try {
        return static_cast<std::uint32_t>(std::stoul(m[1].str()));
    } catch (...) {
        return fallback;
    }
}

Status parse_lower_gate_outcomes(const std::string& payload, std::vector<LowerGateOutcome>* out) {
    if (out == nullptr) {
        return Status::Error("lower gate parse: invalid output pointer");
    }
    out->clear();
    const std::regex row_re(
        "\\{\"parent_top_centroid_numeric_id\":([0-9]+),\"mid_centroid_numeric_id\":([0-9]+),"
        "\"centroid_id\":([0-9]+),\"job_id\":\"([^\"]*)\",\"dataset_size\":([0-9]+),"
        "\"gate_decision\":\"(stop|continue)\"\\}");
    auto it = std::sregex_iterator(payload.begin(), payload.end(), row_re);
    const auto end = std::sregex_iterator();
    for (; it != end; ++it) {
        const std::smatch& m = *it;
        if (m.size() < 7) {
            continue;
        }
        LowerGateOutcome row{};
        try {
            row.branch.parent_top = static_cast<std::uint32_t>(std::stoul(m[1].str()));
            row.branch.mid_centroid = static_cast<std::uint32_t>(std::stoul(m[2].str()));
            row.centroid_id = static_cast<std::uint32_t>(std::stoul(m[3].str()));
            row.job_id = m[4].str();
            row.dataset_size = static_cast<std::uint32_t>(std::stoul(m[5].str()));
            row.gate_decision = m[6].str() == "stop" ? codec::GateDecision::Stop : codec::GateDecision::Continue;
        } catch (...) {
            return Status::Error("lower gate parse: malformed numeric field");
        }
        out->push_back(row);
    }
    const std::uint32_t expected_rows = parse_u32_or_default(payload, "record_count", 0U);
    if (expected_rows != static_cast<std::uint32_t>(out->size())) {
        return Status::Error("lower gate parse: record_count mismatch");
    }
    std::sort(out->begin(), out->end(), [](const LowerGateOutcome& a, const LowerGateOutcome& b) {
        if (a.branch.parent_top != b.branch.parent_top) {
            return a.branch.parent_top < b.branch.parent_top;
        }
        return a.branch.mid_centroid < b.branch.mid_centroid;
    });
    return Status::Ok();
}

Status parse_mid_parent_jobs(const std::string& payload, std::vector<MidParentJob>* out) {
    if (out == nullptr) {
        return Status::Error("mid parent parse: invalid output pointer");
    }
    out->clear();
    const std::regex row_re(
        "\\{\"centroid_id\":([0-9]+),\"job_id\":\"([^\"]*)\",\"dataset_size\":([0-9]+),"
        "\"k_min\":([0-9]+),\"k_max\":([0-9]+),\"chosen_k\":([0-9]+)\\}");
    auto it = std::sregex_iterator(payload.begin(), payload.end(), row_re);
    const auto end = std::sregex_iterator();
    for (; it != end; ++it) {
        const std::smatch& m = *it;
        if (m.size() < 7) {
            continue;
        }
        MidParentJob row{};
        try {
            row.centroid_id = static_cast<std::uint32_t>(std::stoul(m[1].str()));
            row.job_id = m[2].str();
            row.dataset_size = static_cast<std::uint32_t>(std::stoul(m[3].str()));
            row.k_min = static_cast<std::uint32_t>(std::stoul(m[4].str()));
            row.k_max = static_cast<std::uint32_t>(std::stoul(m[5].str()));
            row.chosen_k = static_cast<std::uint32_t>(std::stoul(m[6].str()));
        } catch (...) {
            return Status::Error("mid parent parse: malformed numeric field");
        }
        out->push_back(row);
    }
    return Status::Ok();
}

Status emit_top_layer_artifacts(
    const fs::path& data_dir,
    const std::map<std::uint64_t, Record>& rows,
    std::uint32_t seed,
    const std::function<void(telemetry::EventType, const std::string&, const std::vector<std::pair<std::string, std::string>>&)>& emit_step,
    std::uint32_t* chosen_k_out,
    std::size_t* records_processed_out) {
    if (chosen_k_out == nullptr || records_processed_out == nullptr) {
        return Status::Error("top clustering: invalid output pointers");
    }
    *chosen_k_out = 0U;
    *records_processed_out = 0U;

    std::vector<std::uint64_t> ids;
    std::vector<std::vector<float>> vectors;
    ids.reserve(rows.size());
    vectors.reserve(rows.size());
    for (const auto& kv : rows) {
        if (kv.second.vector.size() != kVectorDim) {
            return Status::Error("top clustering: encountered non-1024D vector");
        }
        ids.push_back(kv.first);
        vectors.push_back(kv.second.vector);
    }
    std::vector<std::vector<float>> kmeans_vectors = vectors;
    if (kmeans_vectors.empty()) {
        kmeans_vectors.push_back(std::vector<float>(kVectorDim, 0.0f));
    }
    const std::uint32_t k_min = derive_k_min(kmeans_vectors.size());
    const std::uint32_t k_max = derive_k_max(kmeans_vectors.size(), k_min);
    codec::IdEstimateRow estimate{};
    estimate.k_min = k_min;
    estimate.k_max = k_max;
    estimate.id_estimate_method = 1U;
    const Status estimate_valid = codec::validate_id_estimate(estimate);
    if (!estimate_valid.ok) {
        return estimate_valid;
    }

    const std::vector<std::uint32_t> coarse = coarse_ks(k_min, k_max);
    std::vector<codec::ElbowTraceRow> elbow_rows;
    elbow_rows.reserve(coarse.size() + 3U);
    std::uint32_t best_k = coarse.front();
    double best_objective = std::numeric_limits<double>::infinity();
    std::vector<std::uint32_t> tested_ks;
    tested_ks.reserve(coarse.size() + 3U);

    for (const std::uint32_t k : coarse) {
        const KMeansResult probe = run_deterministic_kmeans(kmeans_vectors, k, 4U);
        codec::ElbowTraceRow row{};
        row.k_value = k;
        row.objective_value = static_cast<float>(probe.objective);
        row.probe_phase = codec::ProbePhase::Coarse;
        elbow_rows.push_back(row);
        tested_ks.push_back(k);
        if (probe.objective < best_objective ||
            (probe.objective == best_objective && k < best_k)) {
            best_objective = probe.objective;
            best_k = k;
        }
    }

    for (std::uint32_t k = (best_k > k_min ? best_k - 1U : best_k);
         k <= std::min<std::uint32_t>(k_max, best_k + 1U);
         ++k) {
        if (std::find(tested_ks.begin(), tested_ks.end(), k) != tested_ks.end()) {
            continue;
        }
        const KMeansResult probe = run_deterministic_kmeans(kmeans_vectors, k, 4U);
        codec::ElbowTraceRow row{};
        row.k_value = k;
        row.objective_value = static_cast<float>(probe.objective);
        row.probe_phase = codec::ProbePhase::Fine;
        elbow_rows.push_back(row);
        tested_ks.push_back(k);
        if (probe.objective < best_objective ||
            (probe.objective == best_objective && k < best_k)) {
            best_objective = probe.objective;
            best_k = k;
        }
    }

    const Status elbow_valid = codec::validate_elbow_trace(elbow_rows, best_k);
    if (!elbow_valid.ok) {
        return elbow_valid;
    }
    *chosen_k_out = best_k;

    {
        std::ostringstream tested_ks_csv;
        tested_ks_csv << "[";
        for (std::size_t i = 0; i < tested_ks.size(); ++i) {
            if (i > 0) {
                tested_ks_csv << ",";
            }
            tested_ks_csv << tested_ks[i];
        }
        tested_ks_csv << "]";
        emit_step(
            telemetry::EventType::KSelection,
            "running",
            {
                {"k_min", std::to_string(k_min)},
                {"k_max", std::to_string(k_max)},
                {"chosen_k", std::to_string(best_k)},
                {"tested_ks", tested_ks_csv.str()},
            });
    }

    const KMeansResult final_kmeans = run_deterministic_kmeans(kmeans_vectors, best_k, 12U);
    std::vector<codec::TopAssignmentRow> assignments;
    assignments.reserve(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        assignments.push_back(codec::TopAssignmentRow{
            ids[i],
            final_kmeans.assignments[i],
        });
    }
    std::sort(assignments.begin(), assignments.end(), [](const auto& a, const auto& b) {
        return a.embedding_id < b.embedding_id;
    });
    const Status assignment_valid = codec::validate_top_assignments(assignments);
    if (!assignment_valid.ok) {
        return assignment_valid;
    }

    std::vector<codec::TopCentroidRow> centroid_rows;
    centroid_rows.reserve(final_kmeans.centroids.size());
    for (std::uint32_t cid = 0; cid < static_cast<std::uint32_t>(final_kmeans.centroids.size()); ++cid) {
        codec::TopCentroidRow row{};
        row.top_centroid_numeric_id = cid;
        for (std::size_t d = 0; d < kVectorDim; ++d) {
            row.centroid_vector[d] = final_kmeans.centroids[cid][d];
        }
        centroid_rows.push_back(row);
    }
    const Status centroid_valid = codec::validate_top_centroids(centroid_rows);
    if (!centroid_valid.ok) {
        return centroid_valid;
    }

    codec::StabilityReportRow stability{};
    stability.status_code = codec::StabilityStatusCode::Ok;
    stability.reserved = 0U;
    stability.mean_nmi = 1.0f;
    stability.std_nmi = 0.0f;
    stability.mean_jaccard = 1.0f;
    stability.mean_centroid_drift = 0.0f;
    const Status stability_valid = codec::validate_stability_report(stability);
    if (!stability_valid.ok) {
        return stability_valid;
    }

    const fs::path clusters_dir = paths::clusters_current_dir(data_dir);
    fs::create_directories(clusters_dir);

    auto write_and_emit = [&](const fs::path& artifact_path, const Status& status, std::size_t rows_written) {
        std::vector<std::uint8_t> bytes;
        codec::read_file_bytes(artifact_path, &bytes);
        emit_step(
            telemetry::EventType::ArtifactWrite,
            status.ok ? "completed" : "failed",
            {
                {"artifact_path", artifact_path.generic_string()},
                {"rows_written", std::to_string(rows_written)},
                {"bytes_written", std::to_string(bytes.size())},
                {"status", status.ok ? "ok" : "error"},
            });
    };

    Status st = codec::write_id_estimate_file(paths::id_estimate_bin(data_dir), estimate);
    if (!st.ok) {
        return st;
    }
    write_and_emit(paths::id_estimate_bin(data_dir), st, 1U);

    st = codec::write_elbow_trace_file(paths::elbow_trace_bin(data_dir), elbow_rows);
    if (!st.ok) {
        return st;
    }
    write_and_emit(paths::elbow_trace_bin(data_dir), st, elbow_rows.size());

    st = codec::write_top_centroids_file(paths::centroids_bin(data_dir), centroid_rows);
    if (!st.ok) {
        return st;
    }
    write_and_emit(paths::centroids_bin(data_dir), st, centroid_rows.size());

    st = codec::write_top_assignments_file(paths::top_assignments_bin(data_dir), assignments);
    if (!st.ok) {
        return st;
    }
    write_and_emit(paths::top_assignments_bin(data_dir), st, assignments.size());

    st = codec::write_stability_report_file(paths::stability_report_bin(data_dir), stability);
    if (!st.ok) {
        return st;
    }
    write_and_emit(paths::stability_report_bin(data_dir), st, 1U);

    std::vector<std::pair<std::string, std::filesystem::path>> manifest_artifacts = {
        {"id_estimate.bin", paths::id_estimate_bin(data_dir)},
        {"elbow_trace.bin", paths::elbow_trace_bin(data_dir)},
        {"centroids.bin", paths::centroids_bin(data_dir)},
        {"assignments.bin", paths::top_assignments_bin(data_dir)},
        {"stability_report.bin", paths::stability_report_bin(data_dir)},
    };
    std::ostringstream payload_json;
    payload_json << "{";
    payload_json << "\"stage\":\"top\",";
    payload_json << "\"schema_version\":1,";
    payload_json << "\"chosen_k\":" << best_k << ",";
    payload_json << "\"artifacts\":[";
    for (std::size_t i = 0; i < manifest_artifacts.size(); ++i) {
        std::vector<std::uint8_t> artifact_bytes;
        const Status rd = codec::read_file_bytes(manifest_artifacts[i].second, &artifact_bytes);
        if (!rd.ok) {
            return rd;
        }
        if (i > 0) {
            payload_json << ",";
        }
        payload_json << "{";
        payload_json << "\"artifact_path\":\"" << escape_json_string(manifest_artifacts[i].first) << "\",";
        payload_json << "\"artifact_format\":\"binary\",";
        payload_json << "\"endianness\":\"little\",";
        payload_json << "\"record_size_bytes\":0,";
        payload_json << "\"record_count\":0,";
        payload_json << "\"schema_version\":1,";
        payload_json << "\"checksum\":\"" << codec::sha256_hex(artifact_bytes) << "\"";
        payload_json << "}";
    }
    payload_json << "]}";
    const std::string payload = payload_json.str();
    st = codec::write_cluster_manifest_file(
        paths::cluster_manifest_bin(data_dir),
        std::vector<std::uint8_t>(payload.begin(), payload.end()));
    if (!st.ok) {
        return st;
    }
    write_and_emit(paths::cluster_manifest_bin(data_dir), st, 1U);

    *records_processed_out = ids.size();
    (void)seed;
    return Status::Ok();
}

Status emit_mid_layer_artifacts(
    const fs::path& data_dir,
    const std::map<std::uint64_t, Record>& rows,
    std::uint32_t seed,
    const std::function<void(telemetry::EventType, const std::string&, const std::vector<std::pair<std::string, std::string>>&)>& emit_step,
    std::uint32_t* chosen_k_out,
    std::size_t* records_processed_out) {
    if (chosen_k_out == nullptr || records_processed_out == nullptr) {
        return Status::Error("mid clustering: invalid output pointers");
    }
    *chosen_k_out = 0U;
    *records_processed_out = 0U;

    std::vector<codec::TopAssignmentRow> top_assignments;
    const Status read_top = codec::read_top_assignments_file(paths::top_assignments_bin(data_dir), &top_assignments);
    if (!read_top.ok) {
        return Status::Error("mid clustering: failed reading top assignments; run build-top-clusters first");
    }
    const Status top_assignments_valid = codec::validate_top_assignments(top_assignments);
    if (!top_assignments_valid.ok) {
        return Status::Error("mid clustering: invalid top assignments input: " + top_assignments_valid.message);
    }

    struct ParentItem {
        std::uint64_t embedding_id = 0;
        std::vector<float> vector;
    };
    std::map<std::uint32_t, std::vector<ParentItem>> parent_groups;
    parent_groups.clear();
    for (const auto& row : top_assignments) {
        const auto it = rows.find(row.embedding_id);
        if (it == rows.end()) {
            return Status::Error("mid clustering: top assignment references missing live embedding_id");
        }
        if (it->second.vector.size() != kVectorDim) {
            return Status::Error("mid clustering: encountered non-1024D vector");
        }
        parent_groups[row.top_centroid_numeric_id].push_back(ParentItem{row.embedding_id, it->second.vector});
    }

    std::vector<codec::MidAssignmentRow> mid_rows;
    mid_rows.reserve(top_assignments.size());

    struct ParentJobSummary {
        std::uint32_t centroid_id = 0;
        std::string job_id;
        std::uint32_t dataset_size = 0;
        std::uint32_t chosen_k = 0;
        std::uint32_t k_min = 0;
        std::uint32_t k_max = 0;
    };
    std::vector<ParentJobSummary> parent_jobs;
    parent_jobs.reserve(parent_groups.size());

    std::uint32_t global_mid_offset = 0U;
    std::uint32_t parent_job_index = 0U;
    for (const auto& parent_entry : parent_groups) {
        const std::uint32_t parent_id = parent_entry.first;
        const auto& items = parent_entry.second;
        const std::string job_id = "mid-parent-" + std::to_string(parent_job_index++);
        emit_step(
            telemetry::EventType::StageProgress,
            "running",
            {
                {"centroid_id", std::to_string(parent_id)},
                {"job_id", job_id},
                {"job_phase", "start"},
                {"dataset_size", std::to_string(items.size())},
            });

        std::vector<std::vector<float>> parent_vectors;
        parent_vectors.reserve(items.size());
        for (const auto& item : items) {
            parent_vectors.push_back(item.vector);
        }
        if (parent_vectors.empty()) {
            continue;
        }

        const std::uint32_t k_min = derive_k_min(parent_vectors.size());
        const std::uint32_t k_max = derive_k_max(parent_vectors.size(), k_min);
        const std::vector<std::uint32_t> coarse = coarse_ks(k_min, k_max);

        std::uint32_t best_k = coarse.empty() ? k_min : coarse.front();
        double best_objective = std::numeric_limits<double>::infinity();
        std::vector<std::uint32_t> tested_ks;
        tested_ks.reserve(coarse.size() + 3U);

        for (const std::uint32_t k : coarse) {
            const KMeansResult probe = run_deterministic_kmeans(parent_vectors, k, 4U);
            tested_ks.push_back(k);
            if (probe.objective < best_objective ||
                (probe.objective == best_objective && k < best_k)) {
                best_objective = probe.objective;
                best_k = k;
            }
        }
        for (std::uint32_t k = (best_k > k_min ? best_k - 1U : best_k);
             k <= std::min<std::uint32_t>(k_max, best_k + 1U);
             ++k) {
            if (std::find(tested_ks.begin(), tested_ks.end(), k) != tested_ks.end()) {
                continue;
            }
            const KMeansResult probe = run_deterministic_kmeans(parent_vectors, k, 4U);
            tested_ks.push_back(k);
            if (probe.objective < best_objective ||
                (probe.objective == best_objective && k < best_k)) {
                best_objective = probe.objective;
                best_k = k;
            }
        }

        std::ostringstream tested_ks_json;
        tested_ks_json << "[";
        for (std::size_t i = 0; i < tested_ks.size(); ++i) {
            if (i > 0) {
                tested_ks_json << ",";
            }
            tested_ks_json << tested_ks[i];
        }
        tested_ks_json << "]";
        emit_step(
            telemetry::EventType::KSelection,
            "running",
            {
                {"centroid_id", std::to_string(parent_id)},
                {"job_id", job_id},
                {"k_min", std::to_string(k_min)},
                {"k_max", std::to_string(k_max)},
                {"chosen_k", std::to_string(best_k)},
                {"tested_ks", tested_ks_json.str()},
            });

        const KMeansResult final_kmeans = run_deterministic_kmeans(parent_vectors, best_k, 12U);
        for (std::size_t i = 0; i < items.size(); ++i) {
            mid_rows.push_back(codec::MidAssignmentRow{
                items[i].embedding_id,
                static_cast<std::uint32_t>(global_mid_offset + final_kmeans.assignments[i]),
                parent_id,
            });
        }
        parent_jobs.push_back(ParentJobSummary{
            parent_id,
            job_id,
            static_cast<std::uint32_t>(items.size()),
            best_k,
            k_min,
            k_max,
        });
        global_mid_offset += best_k;

        emit_step(
            telemetry::EventType::StageProgress,
            "running",
            {
                {"centroid_id", std::to_string(parent_id)},
                {"job_id", job_id},
                {"job_phase", "end"},
                {"dataset_size", std::to_string(items.size())},
                {"chosen_k", std::to_string(best_k)},
            });
    }

    std::sort(mid_rows.begin(), mid_rows.end(), [](const auto& a, const auto& b) {
        return a.embedding_id < b.embedding_id;
    });
    const Status mid_valid = codec::validate_mid_assignments(mid_rows);
    if (!mid_valid.ok) {
        return mid_valid;
    }
    if (mid_rows.size() != top_assignments.size()) {
        return Status::Error("mid clustering: assignments row-count mismatch vs top assignments");
    }

    const fs::path mid_dir = paths::mid_layer_dir(data_dir);
    fs::create_directories(mid_dir);

    auto write_and_emit = [&](const fs::path& artifact_path, const Status& status, std::size_t rows_written) {
        std::vector<std::uint8_t> bytes;
        codec::read_file_bytes(artifact_path, &bytes);
        emit_step(
            telemetry::EventType::ArtifactWrite,
            status.ok ? "completed" : "failed",
            {
                {"artifact_path", artifact_path.generic_string()},
                {"rows_written", std::to_string(rows_written)},
                {"bytes_written", std::to_string(bytes.size())},
                {"status", status.ok ? "ok" : "error"},
            });
    };

    Status st = codec::write_mid_assignments_file(paths::mid_layer_assignments_bin(data_dir), mid_rows);
    if (!st.ok) {
        return st;
    }
    write_and_emit(paths::mid_layer_assignments_bin(data_dir), st, mid_rows.size());

    std::vector<std::uint8_t> mid_assign_bytes;
    const Status mid_assign_rd = codec::read_file_bytes(paths::mid_layer_assignments_bin(data_dir), &mid_assign_bytes);
    if (!mid_assign_rd.ok) {
        return mid_assign_rd;
    }
    std::ostringstream payload_json;
    payload_json << "{";
    payload_json << "\"stage\":\"mid\",";
    payload_json << "\"schema_version\":1,";
    payload_json << "\"record_count\":" << mid_rows.size() << ",";
    payload_json << "\"artifact_path\":\"mid_layer_clustering/assignments.bin\",";
    payload_json << "\"artifact_format\":\"assignments.bin.v1\",";
    payload_json << "\"endianness\":\"little\",";
    payload_json << "\"record_size_bytes\":" << codec::kMidAssignmentRecordSize << ",";
    payload_json << "\"checksum\":\"" << codec::sha256_hex(mid_assign_bytes) << "\",";
    payload_json << "\"parent_jobs\":[";
    for (std::size_t i = 0; i < parent_jobs.size(); ++i) {
        if (i > 0) {
            payload_json << ",";
        }
        payload_json << "{";
        payload_json << "\"centroid_id\":" << parent_jobs[i].centroid_id << ",";
        payload_json << "\"job_id\":\"" << escape_json_string(parent_jobs[i].job_id) << "\",";
        payload_json << "\"dataset_size\":" << parent_jobs[i].dataset_size << ",";
        payload_json << "\"k_min\":" << parent_jobs[i].k_min << ",";
        payload_json << "\"k_max\":" << parent_jobs[i].k_max << ",";
        payload_json << "\"chosen_k\":" << parent_jobs[i].chosen_k;
        payload_json << "}";
    }
    payload_json << "]}";

    const std::string payload = payload_json.str();
    st = codec::write_cluster_manifest_file(
        paths::mid_layer_clustering_bin(data_dir),
        std::vector<std::uint8_t>(payload.begin(), payload.end()));
    if (!st.ok) {
        return st;
    }
    write_and_emit(paths::mid_layer_clustering_bin(data_dir), st, 1U);

    *chosen_k_out = global_mid_offset;
    *records_processed_out = mid_rows.size();
    (void)seed;
    return Status::Ok();
}

Status emit_lower_layer_artifacts(
    const fs::path& data_dir,
    const std::map<std::uint64_t, Record>& rows,
    std::uint32_t seed,
    const std::function<void(telemetry::EventType, const std::string&, const std::vector<std::pair<std::string, std::string>>&)>& emit_step,
    std::uint32_t* chosen_k_out,
    std::size_t* records_processed_out) {
    if (chosen_k_out == nullptr || records_processed_out == nullptr) {
        return Status::Error("lower clustering: invalid output pointers");
    }
    *chosen_k_out = 0U;
    *records_processed_out = 0U;

    constexpr std::uint32_t kDefaultLowerGateThreshold = 32U;
    std::uint32_t gate_threshold = kDefaultLowerGateThreshold;
    if (const char* env = std::getenv("VECTOR_DB_V3_LOWER_GATE_THRESHOLD")) {
        try {
            const long long parsed = std::stoll(std::string(env));
            if (parsed >= 0 && parsed <= static_cast<long long>(std::numeric_limits<std::uint32_t>::max())) {
                gate_threshold = static_cast<std::uint32_t>(parsed);
            }
        } catch (...) {
            // Keep default threshold on parse errors.
        }
    }

    std::vector<codec::MidAssignmentRow> mid_assignments;
    const Status read_mid = codec::read_mid_assignments_file(paths::mid_layer_assignments_bin(data_dir), &mid_assignments);
    if (!read_mid.ok) {
        return Status::Error("lower clustering: failed reading mid assignments; run build-mid-layer-clusters first");
    }
    const Status mid_valid = codec::validate_mid_assignments(mid_assignments);
    if (!mid_valid.ok) {
        return Status::Error("lower clustering: invalid mid assignments input: " + mid_valid.message);
    }

    struct BranchKey {
        std::uint32_t parent_top = 0;
        std::uint32_t mid_centroid = 0;
        bool operator<(const BranchKey& other) const {
            if (parent_top != other.parent_top) {
                return parent_top < other.parent_top;
            }
            return mid_centroid < other.mid_centroid;
        }
    };
    std::map<BranchKey, std::vector<std::uint64_t>> branch_embedding_ids;
    for (const auto& row : mid_assignments) {
        branch_embedding_ids[BranchKey{row.parent_top_centroid_numeric_id, row.mid_centroid_numeric_id}]
            .push_back(row.embedding_id);
    }

    for (const auto& branch : branch_embedding_ids) {
        if (branch.second.empty()) {
            return Status::Error("lower clustering: encountered empty branch dataset");
        }
    }

    std::size_t stop_count = 0U;
    std::size_t continue_count = 0U;
    std::size_t processed_total = 0U;
    std::ostringstream gate_rows_json;
    gate_rows_json << "[";
    bool first_gate_row = true;
    std::uint32_t branch_index = 0U;
    for (const auto& branch : branch_embedding_ids) {
        const std::string job_id = "lower-branch-" + std::to_string(branch_index++);
        const std::size_t dataset_size = branch.second.size();
        for (const auto embedding_id : branch.second) {
            if (rows.find(embedding_id) == rows.end()) {
                return Status::Error("lower clustering: mid assignment references missing live embedding_id");
            }
        }

        emit_step(
            telemetry::EventType::StageProgress,
            "running",
            {
                {"centroid_id", std::to_string(branch.first.mid_centroid)},
                {"job_id", job_id},
                {"job_phase", "start"},
                {"dataset_size", std::to_string(dataset_size)},
            });

        const codec::GateDecision gate_decision =
            (dataset_size <= gate_threshold) ? codec::GateDecision::Stop : codec::GateDecision::Continue;
        if (gate_decision == codec::GateDecision::Stop) {
            ++stop_count;
        } else {
            ++continue_count;
        }
        processed_total += dataset_size;

        emit_step(
            telemetry::EventType::StageProgress,
            "running",
            {
                {"centroid_id", std::to_string(branch.first.mid_centroid)},
                {"job_id", job_id},
                {"job_phase", "end"},
                {"dataset_size", std::to_string(dataset_size)},
                {"gate_decision", gate_decision == codec::GateDecision::Stop ? "stop" : "continue"},
                {"gate_threshold", std::to_string(gate_threshold)},
            });

        if (!first_gate_row) {
            gate_rows_json << ",";
        }
        first_gate_row = false;
        gate_rows_json << "{";
        gate_rows_json << "\"parent_top_centroid_numeric_id\":" << branch.first.parent_top << ",";
        gate_rows_json << "\"mid_centroid_numeric_id\":" << branch.first.mid_centroid << ",";
        gate_rows_json << "\"centroid_id\":" << branch.first.mid_centroid << ",";
        gate_rows_json << "\"job_id\":\"" << escape_json_string(job_id) << "\",";
        gate_rows_json << "\"dataset_size\":" << dataset_size << ",";
        gate_rows_json << "\"gate_decision\":\"" << (gate_decision == codec::GateDecision::Stop ? "stop" : "continue") << "\"";
        gate_rows_json << "}";
    }
    gate_rows_json << "]";

    const fs::path lower_dir = paths::lower_layer_dir(data_dir);
    fs::create_directories(lower_dir);

    std::ostringstream payload_json;
    payload_json << "{";
    payload_json << "\"stage\":\"lower\",";
    payload_json << "\"schema_version\":1,";
    payload_json << "\"artifact_path\":\"lower_layer_clustering/LOWER_LAYER_CLUSTERING.bin\",";
    payload_json << "\"artifact_format\":\"LOWER_LAYER_CLUSTERING.bin.v1\",";
    payload_json << "\"endianness\":\"little\",";
    payload_json << "\"record_size_bytes\":0,";
    payload_json << "\"record_count\":" << branch_embedding_ids.size() << ",";
    payload_json << "\"gate_threshold\":" << gate_threshold << ",";
    payload_json << "\"rows_processed_total\":" << processed_total << ",";
    payload_json << "\"branches_continue\":" << continue_count << ",";
    payload_json << "\"branches_stop\":" << stop_count << ",";
    payload_json << "\"gate_outcomes\":" << gate_rows_json.str();
    payload_json << "}";

    const std::string payload = payload_json.str();
    Status st = codec::write_cluster_manifest_file(
        paths::lower_layer_clustering_bin(data_dir),
        std::vector<std::uint8_t>(payload.begin(), payload.end()));
    if (!st.ok) {
        return st;
    }

    std::vector<std::uint8_t> lower_bytes;
    codec::read_file_bytes(paths::lower_layer_clustering_bin(data_dir), &lower_bytes);
    emit_step(
        telemetry::EventType::ArtifactWrite,
        "completed",
        {
            {"artifact_path", paths::lower_layer_clustering_bin(data_dir).generic_string()},
            {"rows_written", std::to_string(branch_embedding_ids.size())},
            {"bytes_written", std::to_string(lower_bytes.size())},
            {"status", "ok"},
        });

    *chosen_k_out = static_cast<std::uint32_t>(continue_count + stop_count);
    *records_processed_out = processed_total;
    (void)seed;
    return Status::Ok();
}

Status finalize_k_search_bounds_batch(
    const fs::path& data_dir,
    const std::vector<LowerGateOutcome>& lower_outcomes,
    const std::function<void(telemetry::EventType, const std::string&, const std::vector<std::pair<std::string, std::string>>&)>& emit_step) {
    std::vector<codec::KSearchBoundsBatchRow> rows;

    codec::IdEstimateRow top_estimate{};
    const Status top_estimate_st = codec::read_id_estimate_file(paths::id_estimate_bin(data_dir), &top_estimate);
    if (!top_estimate_st.ok) {
        return Status::Error("finalization: failed reading id_estimate.bin");
    }
    const Status top_estimate_valid = codec::validate_id_estimate(top_estimate);
    if (!top_estimate_valid.ok) {
        return Status::Error("finalization: invalid id_estimate.bin");
    }

    std::vector<codec::TopAssignmentRow> top_assignments;
    const Status top_assign_st = codec::read_top_assignments_file(paths::top_assignments_bin(data_dir), &top_assignments);
    if (!top_assign_st.ok) {
        return Status::Error("finalization: failed reading top assignments");
    }
    const Status top_assign_valid = codec::validate_top_assignments(top_assignments);
    if (!top_assign_valid.ok) {
        return Status::Error("finalization: invalid top assignments");
    }

    std::string top_manifest_payload;
    const Status top_manifest_st = read_manifest_payload_string(paths::cluster_manifest_bin(data_dir), &top_manifest_payload);
    if (!top_manifest_st.ok) {
        return Status::Error("finalization: failed reading top cluster_manifest.bin");
    }
    const std::uint32_t top_chosen_k = parse_u32_or_default(top_manifest_payload, "chosen_k", top_estimate.k_min);
    rows.push_back(codec::KSearchBoundsBatchRow{
        codec::StageLevel::Top,
        codec::GateDecision::NotApplicable,
        0U,
        0U,
        top_estimate.k_min,
        top_estimate.k_max,
        std::max(top_estimate.k_min, std::min(top_chosen_k, top_estimate.k_max)),
        static_cast<std::uint32_t>(top_assignments.size()),
    });

    std::string mid_manifest_payload;
    const Status mid_manifest_st = read_manifest_payload_string(paths::mid_layer_clustering_bin(data_dir), &mid_manifest_payload);
    if (!mid_manifest_st.ok) {
        return Status::Error("finalization: failed reading MID_LAYER_CLUSTERING.bin");
    }
    std::vector<MidParentJob> mid_jobs;
    const Status mid_jobs_st = parse_mid_parent_jobs(mid_manifest_payload, &mid_jobs);
    if (!mid_jobs_st.ok) {
        return Status::Error("finalization: failed parsing MID_LAYER_CLUSTERING.bin");
    }
    for (const auto& job : mid_jobs) {
        rows.push_back(codec::KSearchBoundsBatchRow{
            codec::StageLevel::Mid,
            codec::GateDecision::NotApplicable,
            0U,
            job.centroid_id,
            job.k_min,
            job.k_max,
            job.chosen_k,
            job.dataset_size,
        });
    }

    for (const auto& lower : lower_outcomes) {
        rows.push_back(codec::KSearchBoundsBatchRow{
            codec::StageLevel::Lower,
            lower.gate_decision,
            0U,
            lower.centroid_id,
            1U,
            1U,
            1U,
            lower.dataset_size,
        });
    }

    std::sort(rows.begin(), rows.end(), [](const codec::KSearchBoundsBatchRow& a, const codec::KSearchBoundsBatchRow& b) {
        if (a.stage_level != b.stage_level) {
            return a.stage_level < b.stage_level;
        }
        return a.source_numeric_id < b.source_numeric_id;
    });
    const Status rows_valid = codec::validate_k_search_bounds_batch(rows);
    if (!rows_valid.ok) {
        return Status::Error("finalization: invalid k_search_bounds_batch rows: " + rows_valid.message);
    }
    const Status write_st = codec::write_k_search_bounds_batch_file(paths::k_search_bounds_batch_bin(data_dir), rows);
    if (!write_st.ok) {
        return write_st;
    }

    std::vector<std::uint8_t> bytes;
    codec::read_file_bytes(paths::k_search_bounds_batch_bin(data_dir), &bytes);
    std::size_t rows_top = 0U;
    std::size_t rows_mid = 0U;
    std::size_t rows_lower = 0U;
    for (const auto& row : rows) {
        if (row.stage_level == codec::StageLevel::Top) {
            ++rows_top;
        } else if (row.stage_level == codec::StageLevel::Mid) {
            ++rows_mid;
        } else {
            ++rows_lower;
        }
    }
    emit_step(
        telemetry::EventType::ArtifactWrite,
        "completed",
        {
            {"artifact_path", paths::k_search_bounds_batch_bin(data_dir).generic_string()},
            {"rows_written", std::to_string(rows.size())},
            {"rows_top", std::to_string(rows_top)},
            {"rows_mid", std::to_string(rows_mid)},
            {"rows_lower", std::to_string(rows_lower)},
            {"pipeline_step_name", "finalize_k_search_bounds_batch"},
            {"bytes_written", std::to_string(bytes.size())},
            {"status", "ok"},
        });
    return Status::Ok();
}

Status finalize_post_cluster_membership(
    const fs::path& data_dir,
    const std::map<std::uint64_t, Record>& rows,
    const std::vector<LowerGateOutcome>& lower_outcomes,
    const std::map<std::uint64_t, std::uint32_t>& final_assignment_map,
    const std::function<void(telemetry::EventType, const std::string&, const std::vector<std::pair<std::string, std::string>>&)>& emit_step) {
    std::vector<codec::TopAssignmentRow> top_assignments;
    Status st = codec::read_top_assignments_file(paths::top_assignments_bin(data_dir), &top_assignments);
    if (!st.ok) {
        return Status::Error("finalization: failed reading top assignments");
    }
    st = codec::validate_top_assignments(top_assignments);
    if (!st.ok) {
        return Status::Error("finalization: invalid top assignments");
    }

    std::vector<codec::MidAssignmentRow> mid_assignments;
    st = codec::read_mid_assignments_file(paths::mid_layer_assignments_bin(data_dir), &mid_assignments);
    if (!st.ok) {
        return Status::Error("finalization: failed reading mid assignments");
    }
    st = codec::validate_mid_assignments(mid_assignments);
    if (!st.ok) {
        return Status::Error("finalization: invalid mid assignments");
    }

    std::map<std::uint64_t, std::uint32_t> top_map;
    for (const auto& row : top_assignments) {
        top_map[row.embedding_id] = row.top_centroid_numeric_id;
    }
    struct MidInfo {
        std::uint32_t mid = 0;
        std::uint32_t parent_top = 0;
    };
    std::map<std::uint64_t, MidInfo> mid_map;
    for (const auto& row : mid_assignments) {
        mid_map[row.embedding_id] = MidInfo{row.mid_centroid_numeric_id, row.parent_top_centroid_numeric_id};
    }

    std::map<StageBranchKey, std::uint32_t> lower_centroid_map;
    std::map<StageBranchKey, codec::GateDecision> lower_gate_map;
    for (const auto& row : lower_outcomes) {
        lower_centroid_map[row.branch] = row.centroid_id;
        lower_gate_map[row.branch] = row.gate_decision;
    }

    std::vector<codec::PostClusterMembershipRow> membership_rows;
    membership_rows.reserve(rows.size());
    for (const auto& kv : rows) {
        const std::uint64_t embedding_id = kv.first;
        const auto top_it = top_map.find(embedding_id);
        if (top_it == top_map.end()) {
            return Status::Error("finalization: post membership missing top assignment");
        }
        const auto mid_it = mid_map.find(embedding_id);
        if (mid_it == mid_map.end()) {
            return Status::Error("finalization: post membership missing mid assignment");
        }
        if (mid_it->second.parent_top != top_it->second) {
            return Status::Error("finalization: top/mid parent mismatch");
        }

        const StageBranchKey key{mid_it->second.parent_top, mid_it->second.mid};
        std::uint32_t lower_centroid = std::numeric_limits<std::uint32_t>::max();
        const auto lower_it = lower_centroid_map.find(key);
        if (lower_it != lower_centroid_map.end()) {
            lower_centroid = lower_it->second;
        }

        std::uint32_t final_cluster = std::numeric_limits<std::uint32_t>::max();
        const auto final_it = final_assignment_map.find(embedding_id);
        if (final_it != final_assignment_map.end()) {
            final_cluster = final_it->second;
        } else {
            const auto gate_it = lower_gate_map.find(key);
            if (gate_it != lower_gate_map.end() && gate_it->second == codec::GateDecision::Stop) {
                return Status::Error("finalization: stop-eligible embedding missing final assignment");
            }
        }

        membership_rows.push_back(codec::PostClusterMembershipRow{
            embedding_id,
            top_it->second,
            mid_it->second.mid,
            lower_centroid,
            final_cluster,
        });
    }

    std::sort(membership_rows.begin(), membership_rows.end(), [](const auto& a, const auto& b) {
        return a.embedding_id < b.embedding_id;
    });
    st = codec::validate_post_cluster_membership(membership_rows, true);
    if (!st.ok) {
        return Status::Error("finalization: invalid post_cluster_membership rows: " + st.message);
    }
    if (membership_rows.size() != rows.size()) {
        return Status::Error("finalization: post_cluster_membership row-count mismatch");
    }

    st = codec::write_post_cluster_membership_file(paths::post_cluster_membership_bin(data_dir), membership_rows);
    if (!st.ok) {
        return st;
    }
    std::vector<std::uint8_t> bytes;
    codec::read_file_bytes(paths::post_cluster_membership_bin(data_dir), &bytes);
    emit_step(
        telemetry::EventType::ArtifactWrite,
        "completed",
        {
            {"artifact_path", paths::post_cluster_membership_bin(data_dir).generic_string()},
            {"rows_written", std::to_string(membership_rows.size())},
            {"bytes_written", std::to_string(bytes.size())},
            {"status", "ok"},
        });
    return Status::Ok();
}

Status emit_final_layer_artifacts(
    const fs::path& data_dir,
    const std::map<std::uint64_t, Record>& rows,
    std::uint32_t seed,
    const std::function<void(telemetry::EventType, const std::string&, const std::vector<std::pair<std::string, std::string>>&)>& emit_step,
    std::uint32_t* chosen_k_out,
    std::size_t* records_processed_out) {
    if (chosen_k_out == nullptr || records_processed_out == nullptr) {
        return Status::Error("final clustering: invalid output pointers");
    }
    *chosen_k_out = 0U;
    *records_processed_out = 0U;

    std::vector<codec::MidAssignmentRow> mid_assignments;
    Status st = codec::read_mid_assignments_file(paths::mid_layer_assignments_bin(data_dir), &mid_assignments);
    if (!st.ok) {
        return Status::Error("final clustering: failed reading mid assignments; run build-mid-layer-clusters first");
    }
    st = codec::validate_mid_assignments(mid_assignments);
    if (!st.ok) {
        return Status::Error("final clustering: invalid mid assignments input: " + st.message);
    }

    std::string lower_payload;
    st = read_manifest_payload_string(paths::lower_layer_clustering_bin(data_dir), &lower_payload);
    if (!st.ok) {
        return Status::Error("final clustering: failed reading lower gate outcomes; run build-lower-layer-clusters first");
    }
    std::vector<LowerGateOutcome> lower_outcomes;
    st = parse_lower_gate_outcomes(lower_payload, &lower_outcomes);
    if (!st.ok) {
        return Status::Error("final clustering: failed parsing lower gate outcomes: " + st.message);
    }

    std::map<StageBranchKey, std::vector<std::uint64_t>> branch_embedding_ids;
    for (const auto& row : mid_assignments) {
        branch_embedding_ids[StageBranchKey{row.parent_top_centroid_numeric_id, row.mid_centroid_numeric_id}]
            .push_back(row.embedding_id);
    }

    struct FinalClusterMeta {
        std::uint32_t final_cluster_numeric_id = 0;
        StageBranchKey branch;
        std::string job_id;
        std::size_t dataset_size = 0U;
        std::size_t rows_written = 0U;
    };
    std::vector<FinalClusterMeta> final_clusters;
    std::map<std::uint64_t, std::uint32_t> final_assignment_map;
    std::size_t total_final_rows = 0U;
    std::uint32_t next_final_cluster_id = 0U;
    std::uint32_t final_job_index = 0U;

    auto emit_artifact = [&](const fs::path& artifact_path, std::size_t rows_written) {
        std::vector<std::uint8_t> bytes;
        codec::read_file_bytes(artifact_path, &bytes);
        emit_step(
            telemetry::EventType::ArtifactWrite,
            "completed",
            {
                {"artifact_path", artifact_path.generic_string()},
                {"rows_written", std::to_string(rows_written)},
                {"bytes_written", std::to_string(bytes.size())},
                {"status", "ok"},
            });
    };

    const fs::path final_dir = paths::final_layer_dir(data_dir);
    fs::create_directories(final_dir);

    for (const auto& lower_row : lower_outcomes) {
        if (lower_row.gate_decision != codec::GateDecision::Stop) {
            continue;
        }
        const auto branch_it = branch_embedding_ids.find(lower_row.branch);
        if (branch_it == branch_embedding_ids.end()) {
            return Status::Error("final clustering: stop-eligible lower branch missing in mid assignments");
        }
        const std::vector<std::uint64_t>& ids = branch_it->second;
        if (ids.empty()) {
            return Status::Error("final clustering: stop-eligible lower branch is empty");
        }

        const std::uint32_t final_cluster_id = next_final_cluster_id++;
        const std::string job_id = "final-branch-" + std::to_string(final_job_index++);
        emit_step(
            telemetry::EventType::StageProgress,
            "running",
            {
                {"centroid_id", std::to_string(final_cluster_id)},
                {"job_id", job_id},
                {"job_phase", "start"},
                {"dataset_size", std::to_string(ids.size())},
                {"parent_top_centroid_numeric_id", std::to_string(lower_row.branch.parent_top)},
                {"mid_centroid_numeric_id", std::to_string(lower_row.branch.mid_centroid)},
            });

        std::vector<codec::FinalAssignmentRow> final_rows;
        final_rows.reserve(ids.size());
        for (const std::uint64_t embedding_id : ids) {
            if (rows.find(embedding_id) == rows.end()) {
                return Status::Error("final clustering: mid assignment references missing live embedding_id");
            }
            final_rows.push_back(codec::FinalAssignmentRow{embedding_id, final_cluster_id});
            final_assignment_map[embedding_id] = final_cluster_id;
        }
        std::sort(final_rows.begin(), final_rows.end(), [](const auto& a, const auto& b) {
            return a.embedding_id < b.embedding_id;
        });
        st = codec::validate_final_assignments(final_rows);
        if (!st.ok) {
            return Status::Error("final clustering: invalid final assignments rows: " + st.message);
        }

        st = codec::write_final_assignments_file(paths::final_cluster_assignments_bin(data_dir, final_cluster_id), final_rows);
        if (!st.ok) {
            return st;
        }
        emit_artifact(paths::final_cluster_assignments_bin(data_dir, final_cluster_id), final_rows.size());

        std::vector<std::uint8_t> assignment_bytes;
        st = codec::read_file_bytes(paths::final_cluster_assignments_bin(data_dir, final_cluster_id), &assignment_bytes);
        if (!st.ok) {
            return st;
        }

        std::ostringstream manifest_payload;
        manifest_payload << "{";
        manifest_payload << "\"stage\":\"final\",";
        manifest_payload << "\"pipeline_step_name\":\"final_cluster_write\",";
        manifest_payload << "\"cluster_id\":" << final_cluster_id << ",";
        manifest_payload << "\"job_id\":\"" << escape_json_string(job_id) << "\",";
        manifest_payload << "\"parent_top_centroid_numeric_id\":" << lower_row.branch.parent_top << ",";
        manifest_payload << "\"mid_centroid_numeric_id\":" << lower_row.branch.mid_centroid << ",";
        manifest_payload << "\"artifact_path\":\"final_layer_clustering/final_cluster_" << final_cluster_id << "/assignments.bin\",";
        manifest_payload << "\"artifact_format\":\"assignments.bin.v1\",";
        manifest_payload << "\"endianness\":\"little\",";
        manifest_payload << "\"record_size_bytes\":" << codec::kFinalAssignmentRecordSize << ",";
        manifest_payload << "\"record_count\":" << final_rows.size() << ",";
        manifest_payload << "\"schema_version\":1,";
        manifest_payload << "\"checksum\":\"" << codec::sha256_hex(assignment_bytes) << "\"";
        manifest_payload << "}";
        const std::string manifest_payload_s = manifest_payload.str();
        st = codec::write_cluster_manifest_file(
            paths::final_cluster_manifest_bin(data_dir, final_cluster_id),
            std::vector<std::uint8_t>(manifest_payload_s.begin(), manifest_payload_s.end()));
        if (!st.ok) {
            return st;
        }
        emit_artifact(paths::final_cluster_manifest_bin(data_dir, final_cluster_id), 1U);

        std::ostringstream summary_payload;
        summary_payload << "{";
        summary_payload << "\"stage\":\"final\",";
        summary_payload << "\"cluster_id\":" << final_cluster_id << ",";
        summary_payload << "\"artifact_path\":\"final_layer_clustering/final_cluster_" << final_cluster_id << "/cluster_summary.bin\",";
        summary_payload << "\"artifact_format\":\"cluster_summary.bin.v1\",";
        summary_payload << "\"endianness\":\"little\",";
        summary_payload << "\"record_size_bytes\":0,";
        summary_payload << "\"record_count\":1,";
        summary_payload << "\"schema_version\":1,";
        summary_payload << "\"dataset_size\":" << ids.size() << ",";
        summary_payload << "\"rows_written\":" << final_rows.size() << ",";
        summary_payload << "\"checksum\":\"" << codec::sha256_hex(assignment_bytes) << "\"";
        summary_payload << "}";
        const std::string summary_payload_s = summary_payload.str();
        st = codec::write_cluster_manifest_file(
            paths::final_cluster_summary_bin(data_dir, final_cluster_id),
            std::vector<std::uint8_t>(summary_payload_s.begin(), summary_payload_s.end()));
        if (!st.ok) {
            return st;
        }
        emit_artifact(paths::final_cluster_summary_bin(data_dir, final_cluster_id), 1U);

        emit_step(
            telemetry::EventType::StageProgress,
            "running",
            {
                {"centroid_id", std::to_string(final_cluster_id)},
                {"job_id", job_id},
                {"job_phase", "end"},
                {"dataset_size", std::to_string(ids.size())},
                {"parent_top_centroid_numeric_id", std::to_string(lower_row.branch.parent_top)},
                {"mid_centroid_numeric_id", std::to_string(lower_row.branch.mid_centroid)},
            });

        total_final_rows += final_rows.size();
        final_clusters.push_back(FinalClusterMeta{
            final_cluster_id,
            lower_row.branch,
            job_id,
            ids.size(),
            final_rows.size(),
        });
    }

    std::ostringstream aggregate_payload;
    aggregate_payload << "{";
    aggregate_payload << "\"stage\":\"final\",";
    aggregate_payload << "\"schema_version\":1,";
    aggregate_payload << "\"artifact_path\":\"final_layer_clustering/FINAL_LAYER_CLUSTERS.bin\",";
    aggregate_payload << "\"artifact_format\":\"FINAL_LAYER_CLUSTERS.bin.v1\",";
    aggregate_payload << "\"endianness\":\"little\",";
    aggregate_payload << "\"record_size_bytes\":0,";
    aggregate_payload << "\"record_count\":" << final_clusters.size() << ",";
    aggregate_payload << "\"eligible_branches_stop\":" << final_clusters.size() << ",";
    aggregate_payload << "\"rows_processed_total\":" << total_final_rows << ",";
    aggregate_payload << "\"clusters\":[";
    for (std::size_t i = 0; i < final_clusters.size(); ++i) {
        if (i > 0) {
            aggregate_payload << ",";
        }
        const auto& c = final_clusters[i];
        aggregate_payload << "{";
        aggregate_payload << "\"final_cluster_numeric_id\":" << c.final_cluster_numeric_id << ",";
        aggregate_payload << "\"job_id\":\"" << escape_json_string(c.job_id) << "\",";
        aggregate_payload << "\"parent_top_centroid_numeric_id\":" << c.branch.parent_top << ",";
        aggregate_payload << "\"mid_centroid_numeric_id\":" << c.branch.mid_centroid << ",";
        aggregate_payload << "\"dataset_size\":" << c.dataset_size << ",";
        aggregate_payload << "\"rows_written\":" << c.rows_written;
        aggregate_payload << "}";
    }
    aggregate_payload << "]}";
    const std::string aggregate_payload_s = aggregate_payload.str();
    st = codec::write_cluster_manifest_file(
        paths::final_layer_clusters_bin(data_dir),
        std::vector<std::uint8_t>(aggregate_payload_s.begin(), aggregate_payload_s.end()));
    if (!st.ok) {
        return st;
    }
    emit_artifact(paths::final_layer_clusters_bin(data_dir), final_clusters.size());

    st = finalize_k_search_bounds_batch(data_dir, lower_outcomes, emit_step);
    if (!st.ok) {
        return st;
    }
    st = finalize_post_cluster_membership(data_dir, rows, lower_outcomes, final_assignment_map, emit_step);
    if (!st.ok) {
        return st;
    }

    *chosen_k_out = static_cast<std::uint32_t>(final_clusters.size());
    *records_processed_out = total_final_rows;
    (void)seed;
    return Status::Ok();
}

}  // namespace

struct VectorStore::Impl {
    explicit Impl(std::string d) : data_dir(std::move(d)) {}

    fs::path data_dir;
    bool opened = false;
    std::map<std::uint64_t, Record> rows;
    std::uint64_t last_lsn = 0;
    std::uint64_t checkpoint_lsn = 0;
    std::size_t wal_entries = 0;
    std::size_t tombstone_rows = 0;
    std::string checkpoint_file;
};

VectorStore::VectorStore(std::string data_dir) : impl_(new Impl(std::move(data_dir))) {}
VectorStore::~VectorStore() { delete impl_; }

Status VectorStore::init() {
    try {
        fs::create_directories(impl_->data_dir);
        fs::create_directories(paths::segments_dir(impl_->data_dir));
        fs::create_directories(paths::clusters_current_dir(impl_->data_dir));
        if (!fs::exists(paths::manifest(impl_->data_dir))) {
            ManifestMeta meta{};
            const Status write = write_manifest_json_atomic(paths::manifest(impl_->data_dir), meta);
            if (!write.ok) {
                return write;
            }
        }
        if (!fs::exists(paths::wal(impl_->data_dir))) {
            const Status write = codec::write_atomic_bytes(paths::wal(impl_->data_dir), {});
            if (!write.ok) {
                return Status::Error("failed creating wal.log: " + write.message);
            }
        }
        return Status::Ok();
    } catch (const std::exception& e) {
        return Status::Error(std::string("init failed: ") + e.what());
    } catch (...) {
        return Status::Error("init failed");
    }
}

Status VectorStore::open() {
    if (!fs::exists(paths::manifest(impl_->data_dir))) {
        return Status::Error("manifest.json missing; run init first");
    }
    if (!fs::exists(paths::wal(impl_->data_dir))) {
        const Status make_wal = codec::write_atomic_bytes(paths::wal(impl_->data_dir), {});
        if (!make_wal.ok) {
            return Status::Error("failed creating wal.log on open: " + make_wal.message);
        }
    }
    impl_->rows.clear();
    impl_->last_lsn = 0;
    impl_->checkpoint_lsn = 0;
    impl_->wal_entries = 0;
    impl_->tombstone_rows = 0;
    impl_->checkpoint_file.clear();

    ManifestMeta manifest{};
    const Status m = load_manifest_json(paths::manifest(impl_->data_dir), &manifest);
    if (!m.ok) {
        return m;
    }
    impl_->checkpoint_lsn = manifest.checkpoint_lsn;
    impl_->last_lsn = manifest.last_lsn;
    impl_->checkpoint_file = manifest.checkpoint_file;
    if (!manifest.checkpoint_file.empty()) {
        const fs::path cp = impl_->data_dir / manifest.checkpoint_file;
        if (!fs::exists(cp)) {
            return Status::Error("checkpoint file missing: " + cp.string());
        }
        std::uint64_t cp_lsn = 0;
        const Status load_cp = load_checkpoint_snapshot(cp, &cp_lsn, &impl_->rows);
        if (!load_cp.ok) {
            return load_cp;
        }
        if (cp_lsn != manifest.checkpoint_lsn) {
            return Status::Error("checkpoint_lsn mismatch between manifest and checkpoint file");
        }
    }
    const Status replay = replay_wal_entries(
        paths::wal(impl_->data_dir),
        impl_->checkpoint_lsn,
        &impl_->last_lsn,
        &impl_->wal_entries,
        &impl_->tombstone_rows,
        &impl_->rows);
    if (!replay.ok) {
        return replay;
    }
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
    const std::uint64_t snapshot_lsn = impl_->last_lsn;
    const fs::path checkpoint_path = paths::checkpoint_bin(impl_->data_dir, snapshot_lsn);
    const Status write_cp = write_checkpoint_snapshot(checkpoint_path, snapshot_lsn, impl_->rows);
    if (!write_cp.ok) {
        return write_cp;
    }

    ManifestMeta meta{};
    meta.checkpoint_lsn = snapshot_lsn;
    meta.last_lsn = impl_->last_lsn;
    meta.checkpoint_file = fs::relative(checkpoint_path, impl_->data_dir).generic_string();
    const Status write_manifest = write_manifest_json_atomic(paths::manifest(impl_->data_dir), meta);
    if (!write_manifest.ok) {
        return write_manifest;
    }

    const Status wal_reset = codec::write_atomic_bytes(paths::wal(impl_->data_dir), {});
    if (!wal_reset.ok) {
        return Status::Error("checkpoint succeeded but wal rotation failed: " + wal_reset.message);
    }
    impl_->checkpoint_lsn = snapshot_lsn;
    impl_->wal_entries = 0;
    impl_->checkpoint_file = meta.checkpoint_file;
    impl_->tombstone_rows = 0;
    return Status::Ok();
}

Status VectorStore::insert(std::uint64_t embedding_id, const std::vector<float>& vector_fp32_1024) {
    if (!impl_->opened) {
        return Status::Error("store not open");
    }
    if (vector_fp32_1024.size() != kVectorDim) {
        return Status::Error("insert: vector dimension mismatch");
    }
    WalEntry entry{};
    entry.lsn = impl_->last_lsn + 1U;
    entry.op = kWalOpInsert;
    entry.embedding_id = embedding_id;
    entry.vector = vector_fp32_1024;
    const Status wal = append_wal_entry(paths::wal(impl_->data_dir), entry);
    if (!wal.ok) {
        return wal;
    }
    impl_->rows[embedding_id] = Record{embedding_id, vector_fp32_1024};
    impl_->last_lsn = entry.lsn;
    impl_->wal_entries += 1U;
    return Status::Ok();
}

Status VectorStore::insert_batch(const std::vector<Record>& records) {
    if (!impl_->opened) {
        return Status::Error("store not open");
    }
    for (const auto& rec : records) {
        const Status s = insert(rec.embedding_id, rec.vector);
        if (!s.ok) {
            return s;
        }
    }
    return Status::Ok();
}

Status VectorStore::remove(std::uint64_t embedding_id) {
    if (!impl_->opened) {
        return Status::Error("store not open");
    }
    const auto it = impl_->rows.find(embedding_id);
    if (it == impl_->rows.end()) {
        return Status::Error("delete: embedding_id not found");
    }
    WalEntry entry{};
    entry.lsn = impl_->last_lsn + 1U;
    entry.op = kWalOpDelete;
    entry.embedding_id = embedding_id;
    const Status wal = append_wal_entry(paths::wal(impl_->data_dir), entry);
    if (!wal.ok) {
        return wal;
    }
    impl_->rows.erase(it);
    impl_->last_lsn = entry.lsn;
    impl_->wal_entries += 1U;
    impl_->tombstone_rows += 1U;
    return Status::Ok();
}

std::optional<Record> VectorStore::get(std::uint64_t embedding_id) const {
    const auto it = impl_->rows.find(embedding_id);
    if (it == impl_->rows.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::vector<SearchResult> VectorStore::search_exact(const std::vector<float>& query, std::size_t top_k) const {
    if (query.size() != kVectorDim || top_k == 0 || impl_->rows.empty()) {
        return {};
    }

    std::vector<SearchResult> out;
    out.reserve(impl_->rows.size());
    for (const auto& kv : impl_->rows) {
        const std::uint64_t embedding_id = kv.first;
        const auto& row = kv.second;
        if (row.vector.size() != kVectorDim) {
            continue;
        }
        double score = 0.0;
        for (std::size_t i = 0; i < kVectorDim; ++i) {
            score += static_cast<double>(query[i]) * static_cast<double>(row.vector[i]);
        }
        if (!std::isfinite(score)) {
            score = -std::numeric_limits<double>::infinity();
        }
        out.push_back(SearchResult{embedding_id, score});
    }

    std::sort(out.begin(), out.end(), [](const SearchResult& a, const SearchResult& b) {
        if (a.score > b.score) {
            return true;
        }
        if (a.score < b.score) {
            return false;
        }
        return a.embedding_id < b.embedding_id;
    });

    if (top_k < out.size()) {
        out.resize(top_k);
    }
    return out;
}

Stats VectorStore::stats() const {
    Stats out{};
    out.total_rows = impl_->rows.size() + impl_->tombstone_rows;
    out.live_rows = impl_->rows.size();
    out.tombstone_rows = impl_->tombstone_rows;
    return out;
}

WalStats VectorStore::wal_stats() const {
    WalStats out{};
    out.checkpoint_lsn = impl_->checkpoint_lsn;
    out.last_lsn = impl_->last_lsn;
    out.wal_entries = impl_->wal_entries;
    return out;
}

ClusterStats VectorStore::cluster_stats() const {
    ClusterStats out{};
    out.available = false;
    out.compliance_status = "fail";
#if VECTOR_DB_V3_CUDA_ENABLED
    out.cuda_enabled = true;
    out.tensor_core_active = true;
    out.gpu_arch_class = "ampere";
    out.kernel_backend_path = "cuda_scaffold";
    out.compliance_status = "pass";
#else
    out.cuda_enabled = false;
    out.tensor_core_active = false;
    out.gpu_arch_class = "unknown";
    out.kernel_backend_path = "none";
    out.fallback_reason = "scaffold_no_cuda_runtime";
#endif
    return out;
}

ClusterHealth VectorStore::cluster_health() const {
    ClusterHealth out{};
    out.available = true;
    out.passed = true;
    out.status = "section2_scaffold";
    return out;
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

Status run_stage_with_telemetry(
    const fs::path& data_dir,
    const std::string& stage_id,
    const std::string& stage_name,
    const std::map<std::uint64_t, Record>& rows,
    std::uint32_t seed) {
    using steady_clock = std::chrono::steady_clock;
    const auto pipeline_start = steady_clock::now();
    const std::string pipeline_started_ts = telemetry::now_ts();
    double last_pipeline_elapsed_ms = 0.0;
    std::unordered_map<std::string, double> current_stage_elapsed_ms;

    std::unordered_map<std::string, double> previous_stage_elapsed_ms;
    bool previous_run_available = false;
    std::string baseline_unavailable_reason;
    const fs::path baseline_path = paths::telemetry_baseline_jsonl(data_dir);
    const Status baseline_load = telemetry::load_stage_baseline(
        baseline_path,
        &previous_stage_elapsed_ms,
        &previous_run_available,
        &baseline_unavailable_reason);
    if (!baseline_load.ok) {
        previous_run_available = false;
        baseline_unavailable_reason = "baseline_load_error";
    }

    auto pipeline_elapsed_now_ms = [&]() -> double {
        const auto now = steady_clock::now();
        const auto elapsed = std::chrono::duration<double, std::milli>(now - pipeline_start).count();
        return telemetry::monotonic_ms(elapsed, &last_pipeline_elapsed_ms);
    };

    auto stage_prev_extra = [&]() -> std::vector<std::pair<std::string, std::string>> {
        std::vector<std::pair<std::string, std::string>> extra;
        const auto prev_it = previous_stage_elapsed_ms.find(stage_id);
        const bool has_prev = previous_run_available && prev_it != previous_stage_elapsed_ms.end();
        extra.push_back({"previous_run_available", has_prev ? "true" : "false"});
        if (has_prev) {
            std::ostringstream os_prev;
            os_prev << std::fixed << std::setprecision(3) << prev_it->second;
            extra.push_back({"previous_run_stage_elapsed_ms", os_prev.str()});
        } else {
            extra.push_back({"previous_run_stage_elapsed_ms", "null"});
            if (!previous_run_available && !baseline_unavailable_reason.empty()) {
                extra.push_back({"baseline_unavailable_reason", baseline_unavailable_reason});
            }
        }
        return extra;
    };

    telemetry::emit_event(
        std::cout,
        telemetry::EventType::PipelineStart,
        "pipeline",
        "Pipeline",
        "running",
        pipeline_started_ts,
        std::nullopt,
        0.0,
        0.0,
        "running",
        stage_prev_extra());

    const auto stage_start_steady = steady_clock::now();
    const std::string stage_started_ts = telemetry::now_ts();
    {
        std::vector<std::pair<std::string, std::string>> extra = stage_prev_extra();
        extra.push_back({"stage_started_ts", stage_started_ts});
        extra.push_back({"stage_elapsed_ms", "0.000"});
        telemetry::emit_event(
            std::cout,
            telemetry::EventType::StageStart,
            stage_id,
            stage_name,
            "running",
            stage_started_ts,
            std::nullopt,
            0.0,
            pipeline_elapsed_now_ms(),
            "running",
            extra);
    }

    const bool force_skip = env_truthy(std::getenv("VECTOR_DB_V3_FORCE_STAGE_SKIP"));
    const bool force_fail = env_truthy(std::getenv("VECTOR_DB_V3_FORCE_STAGE_FAIL"));
    const bool force_compliance_fail = env_truthy(std::getenv("VECTOR_DB_V3_FORCE_COMPLIANCE_FAIL"));

    std::string final_status = "completed";
    telemetry::EventType terminal_event_type = telemetry::EventType::StageEnd;
    Status stage_status = Status::Ok();
    std::vector<std::pair<std::string, std::string>> terminal_extra = stage_prev_extra();
    std::size_t records_processed = 0;
    std::uint32_t chosen_k = 0U;

    if (force_skip) {
        final_status = "skipped";
        terminal_event_type = telemetry::EventType::StageSkip;
    } else if (force_compliance_fail) {
        final_status = "failed";
        terminal_event_type = telemetry::EventType::StageFail;
        stage_status = Status::Error("forced compliance failure");
        terminal_extra.push_back({"error_code", "compliance_fail_fast"});
        terminal_extra.push_back({"error_message", "forced compliance failure"});
        terminal_extra.push_back({"non_compliance_stage", stage_id});
        terminal_extra.push_back({"compliance_status", "fail"});
    } else if (force_fail) {
        final_status = "failed";
        terminal_event_type = telemetry::EventType::StageFail;
        stage_status = Status::Error("forced runtime failure");
        terminal_extra.push_back({"error_code", "runtime_error"});
        terminal_extra.push_back({"error_message", "forced runtime failure"});
    } else {
        if (stage_id == "top") {
            auto emit_step = [&](telemetry::EventType step_type,
                                 const std::string& step_status,
                                 const std::vector<std::pair<std::string, std::string>>& extra) {
                telemetry::emit_event(
                    std::cout,
                    step_type,
                    stage_id,
                    stage_name,
                    step_status,
                    stage_started_ts,
                    std::nullopt,
                    0.0,
                    pipeline_elapsed_now_ms(),
                    "running",
                    extra);
            };
            stage_status = emit_top_layer_artifacts(
                data_dir,
                rows,
                seed,
                emit_step,
                &chosen_k,
                &records_processed);
            if (!stage_status.ok) {
                final_status = "failed";
                terminal_event_type = telemetry::EventType::StageFail;
                terminal_extra.push_back({"error_code", "top_layer_build_failed"});
                terminal_extra.push_back({"error_message", stage_status.message});
            }
        } else if (stage_id == "mid") {
            auto emit_step = [&](telemetry::EventType step_type,
                                 const std::string& step_status,
                                 const std::vector<std::pair<std::string, std::string>>& extra) {
                telemetry::emit_event(
                    std::cout,
                    step_type,
                    stage_id,
                    stage_name,
                    step_status,
                    stage_started_ts,
                    std::nullopt,
                    0.0,
                    pipeline_elapsed_now_ms(),
                    "running",
                    extra);
            };
            stage_status = emit_mid_layer_artifacts(
                data_dir,
                rows,
                seed,
                emit_step,
                &chosen_k,
                &records_processed);
            if (!stage_status.ok) {
                final_status = "failed";
                terminal_event_type = telemetry::EventType::StageFail;
                terminal_extra.push_back({"error_code", "mid_layer_build_failed"});
                terminal_extra.push_back({"error_message", stage_status.message});
            }
        } else if (stage_id == "lower") {
            auto emit_step = [&](telemetry::EventType step_type,
                                 const std::string& step_status,
                                 const std::vector<std::pair<std::string, std::string>>& extra) {
                telemetry::emit_event(
                    std::cout,
                    step_type,
                    stage_id,
                    stage_name,
                    step_status,
                    stage_started_ts,
                    std::nullopt,
                    0.0,
                    pipeline_elapsed_now_ms(),
                    "running",
                    extra);
            };
            stage_status = emit_lower_layer_artifacts(
                data_dir,
                rows,
                seed,
                emit_step,
                &chosen_k,
                &records_processed);
            if (!stage_status.ok) {
                final_status = "failed";
                terminal_event_type = telemetry::EventType::StageFail;
                terminal_extra.push_back({"error_code", "lower_layer_build_failed"});
                terminal_extra.push_back({"error_message", stage_status.message});
            }
        } else if (stage_id == "final") {
            auto emit_step = [&](telemetry::EventType step_type,
                                 const std::string& step_status,
                                 const std::vector<std::pair<std::string, std::string>>& extra) {
                telemetry::emit_event(
                    std::cout,
                    step_type,
                    stage_id,
                    stage_name,
                    step_status,
                    stage_started_ts,
                    std::nullopt,
                    0.0,
                    pipeline_elapsed_now_ms(),
                    "running",
                    extra);
            };
            stage_status = emit_final_layer_artifacts(
                data_dir,
                rows,
                seed,
                emit_step,
                &chosen_k,
                &records_processed);
            if (!stage_status.ok) {
                final_status = "failed";
                terminal_event_type = telemetry::EventType::StageFail;
                terminal_extra.push_back({"error_code", "final_layer_build_failed"});
                terminal_extra.push_back({"error_message", stage_status.message});
            }
        }
        terminal_extra.push_back({"records_processed", std::to_string(records_processed)});
        terminal_extra.push_back({"chosen_k", std::to_string(chosen_k)});
        terminal_extra.push_back({"compliance_status", stage_status.ok ? "pass" : "fail"});
    }

    const auto stage_end_steady = steady_clock::now();
    const double raw_stage_elapsed_ms =
        std::chrono::duration<double, std::milli>(stage_end_steady - stage_start_steady).count();
    const double stage_elapsed_ms = raw_stage_elapsed_ms >= 0.0 && std::isfinite(raw_stage_elapsed_ms) ?
        raw_stage_elapsed_ms : 0.0;
    current_stage_elapsed_ms[stage_id] = stage_elapsed_ms;

    {
        std::ostringstream stage_elapsed_os;
        stage_elapsed_os << std::fixed << std::setprecision(3) << stage_elapsed_ms;
        terminal_extra.push_back({"stage_started_ts", stage_started_ts});
        terminal_extra.push_back({"stage_elapsed_ms", stage_elapsed_os.str()});
    }
    telemetry::emit_event(
        std::cout,
        terminal_event_type,
        stage_id,
        stage_name,
        final_status,
        stage_started_ts,
        telemetry::now_ts(),
        stage_elapsed_ms,
        pipeline_elapsed_now_ms(),
        final_status,
        terminal_extra);

    std::vector<std::pair<std::string, std::string>> summary_extra = {
        {"stages_completed", stage_status.ok && !force_skip ? "1" : "0"},
        {"stages_failed", stage_status.ok ? "0" : "1"},
        {"stages_skipped", force_skip ? "1" : "0"},
        {"records_processed_total", std::to_string(records_processed)},
        {"final_output_status", stage_status.ok ? (force_skip ? "skipped" : "success") : "failed"},
        {"summary_version", "1"},
    };
    if (!previous_run_available) {
        summary_extra.push_back({"baseline_unavailable_reason",
                                 baseline_unavailable_reason.empty() ? "no_previous_run_baseline" :
                                     baseline_unavailable_reason});
    }

    telemetry::emit_event(
        std::cout,
        telemetry::EventType::PipelineSummary,
        "pipeline",
        "Pipeline",
        stage_status.ok ? "completed" : "failed",
        pipeline_started_ts,
        telemetry::now_ts(),
        pipeline_elapsed_now_ms(),
        pipeline_elapsed_now_ms(),
        stage_status.ok ? "completed" : "failed",
        summary_extra);

    const Status baseline_write = telemetry::write_stage_baseline(baseline_path, current_stage_elapsed_ms);
    if (!baseline_write.ok && stage_status.ok) {
        return baseline_write;
    }
    return stage_status;
}

Status VectorStore::build_top_clusters(std::uint32_t seed) {
    if (!impl_->opened) {
        return Status::Error("store not open");
    }
    return run_stage_with_telemetry(impl_->data_dir, "top", "Top Layer", impl_->rows, seed);
}

Status VectorStore::build_mid_layer_clusters(std::uint32_t seed) {
    if (!impl_->opened) {
        return Status::Error("store not open");
    }
    return run_stage_with_telemetry(impl_->data_dir, "mid", "Mid Layer", impl_->rows, seed);
}

Status VectorStore::build_lower_layer_clusters(std::uint32_t seed) {
    if (!impl_->opened) {
        return Status::Error("store not open");
    }
    return run_stage_with_telemetry(impl_->data_dir, "lower", "Lower Layer", impl_->rows, seed);
}

Status VectorStore::build_final_layer_clusters(std::uint32_t seed) {
    if (!impl_->opened) {
        return Status::Error("store not open");
    }
    return run_stage_with_telemetry(impl_->data_dir, "final", "Final Layer", impl_->rows, seed);
}

}  // namespace vector_db_v3
