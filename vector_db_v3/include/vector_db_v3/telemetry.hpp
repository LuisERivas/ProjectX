#pragma once

#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace vector_db_v3::telemetry {

std::string now_ts();
std::string json_escape(const std::string& in);

void emit_event(
    std::ostream& out,
    const std::string& event_type,
    const std::string& stage_id,
    const std::string& stage_name,
    const std::string& status,
    double elapsed_ms,
    double pipeline_elapsed_ms,
    const std::vector<std::pair<std::string, std::string>>& extra = {});

}  // namespace vector_db_v3::telemetry
