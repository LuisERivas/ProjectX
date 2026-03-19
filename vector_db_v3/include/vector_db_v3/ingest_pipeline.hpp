#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "vector_db_v3/status.hpp"
#include "vector_db_v3/vector_store.hpp"

namespace vector_db_v3::ingest {

struct PipelineOptions {
    std::size_t batch_size = 1000U;
    std::size_t queue_capacity_batches = 4U;
    bool async_enabled = false;
    bool request_pinned = false;
};

struct PipelineStats {
    bool async_enabled = false;
    bool pinned_enabled = false;
    std::string pinned_mode = "off";
    std::size_t batches_committed = 0U;
    std::size_t records_committed = 0U;
};

using RecordBatch = std::vector<Record>;
using BatchProducer = std::function<Status(RecordBatch* out_batch, bool* eof)>;

Status run_pipeline(
    VectorStore* store,
    const PipelineOptions& options,
    const BatchProducer& producer,
    PipelineStats* stats_out);

}  // namespace vector_db_v3::ingest
