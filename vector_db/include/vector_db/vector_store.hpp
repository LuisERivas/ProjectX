#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vector_db/status.hpp"

namespace vector_db {

constexpr std::size_t kVectorDim = 1024;

struct Record {
    std::uint64_t id = 0;
    std::vector<float> vector_fp32;
    std::string metadata_json;
};

struct StoredRecord {
    std::uint64_t id = 0;
    bool deleted = false;
    std::vector<float> vector_fp32;
    std::string metadata_json;
};

struct DirtyRange {
    std::uint64_t segment_id = 1;
    std::size_t start_row = 0;
    std::size_t end_row = 0;
    std::string reason;
};

struct Stats {
    std::size_t dimension = kVectorDim;
    std::size_t total_rows = 0;
    std::size_t live_rows = 0;
    std::size_t tombstone_rows = 0;
    std::size_t segments = 0;
    std::size_t dirty_ranges = 0;
};

class VectorStore {
public:
    explicit VectorStore(std::string data_dir);
    ~VectorStore();

    Status init();
    Status open();
    Status flush();
    Status close();

    Status insert(std::uint64_t id, const std::vector<float>& vector_fp32_1024, const std::string& metadata_json, bool upsert = false);
    Status remove(std::uint64_t id);
    Status update_metadata(std::uint64_t id, const std::string& patch_json);
    std::optional<StoredRecord> get(std::uint64_t id) const;
    Stats stats() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace vector_db

