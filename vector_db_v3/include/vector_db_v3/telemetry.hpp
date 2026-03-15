#pragma once

#include <filesystem>
#include <ostream>
#include <string>
#include <unordered_map>
#include <optional>
#include <utility>
#include <vector>

#include "vector_db_v3/status.hpp"

namespace vector_db_v3::telemetry {

std::string now_ts();
std::string json_escape(const std::string& in);

enum class EventType {
    PipelineStart,
    StageStart,
    StageEnd,
    StageFail,
    StageSkip,
    PipelineSummary,
};

std::string event_type_to_string(EventType event_type);

double monotonic_ms(double candidate_ms, double* last_ms);

Status load_stage_baseline(
    const std::filesystem::path& path,
    std::unordered_map<std::string, double>* out,
    bool* previous_run_available,
    std::string* unavailable_reason);

Status write_stage_baseline(
    const std::filesystem::path& path,
    const std::unordered_map<std::string, double>& stage_elapsed_ms);

void emit_event(
    std::ostream& out,
    EventType event_type,
    const std::string& stage_id,
    const std::string& stage_name,
    const std::string& status,
    const std::string& start_ts,
    const std::optional<std::string>& end_ts,
    double elapsed_ms,
    double pipeline_elapsed_ms,
    const std::string& active_pipeline_state,
    const std::vector<std::pair<std::string, std::string>>& extra = {});

}  // namespace vector_db_v3::telemetry
