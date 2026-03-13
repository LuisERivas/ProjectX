#pragma once

#include <string>
#include <utility>

namespace vector_db_v2 {

struct Status {
    bool ok = true;
    std::string message;

    static Status Ok() { return {true, ""}; }
    static Status Error(std::string msg) { return {false, std::move(msg)}; }
};

}  // namespace vector_db_v2
