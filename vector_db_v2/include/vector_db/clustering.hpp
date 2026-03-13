#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vector_db_v2 {

struct AssignmentRow {
    std::uint64_t embedding_id = 0;
    std::string centroid_id;
};

struct GateDecisionRow {
    std::string centroid_id;
    std::string decision;  // continue|stop
    std::size_t dataset_size = 0;
    std::string reason;
};

struct DbscanLabelRow {
    std::uint64_t embedding_id = 0;
    int label = -1;
};

}  // namespace vector_db_v2
