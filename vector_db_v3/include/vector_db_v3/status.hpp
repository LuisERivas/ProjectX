#pragma once

#include <string>
#include <utility>

namespace vector_db_v3 {

struct Status {
    bool ok = false;
    std::string message;
    int code = 1;

    static Status Ok() { return Status{true, "", 0}; }
    static Status Error(std::string msg, int c = 1) { return Status{false, std::move(msg), c}; }
};

}  // namespace vector_db_v3
