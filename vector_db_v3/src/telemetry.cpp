#include "vector_db_v3/telemetry.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "vector_db_v3/codec/io.hpp"

namespace vector_db_v3::telemetry {

std::string now_ts() {
    const auto now = std::chrono::system_clock::now();
    const auto secs = std::chrono::duration<double>(now.time_since_epoch()).count();
    std::ostringstream os;
    os << std::fixed << std::setprecision(3) << secs;
    return os.str();
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

std::string event_type_to_string(const EventType event_type) {
    switch (event_type) {
        case EventType::PipelineStart:
            return "pipeline_start";
        case EventType::StageStart:
            return "stage_start";
        case EventType::StageEnd:
            return "stage_end";
        case EventType::StageFail:
            return "stage_fail";
        case EventType::StageSkip:
            return "stage_skip";
        case EventType::PipelineSummary:
            return "pipeline_summary";
        case EventType::StageProgress:
            return "stage_progress";
        case EventType::ArtifactWrite:
            return "artifact_write";
        case EventType::KSelection:
            return "k_selection";
        case EventType::ComplianceCheck:
            return "compliance_check";
    }
    return "unknown";
}

double monotonic_ms(double candidate_ms, double* last_ms) {
    if (!std::isfinite(candidate_ms) || candidate_ms < 0.0) {
        candidate_ms = 0.0;
    }
    if (last_ms == nullptr) {
        return candidate_ms;
    }
    if (candidate_ms < *last_ms) {
        candidate_ms = *last_ms;
    } else {
        *last_ms = candidate_ms;
    }
    return candidate_ms;
}

Status load_stage_baseline(
    const std::filesystem::path& path,
    std::unordered_map<std::string, double>* out,
    bool* previous_run_available,
    std::string* unavailable_reason) {
    if (out == nullptr || previous_run_available == nullptr || unavailable_reason == nullptr) {
        return Status::Error("load_stage_baseline: invalid output pointers");
    }
    out->clear();
    *previous_run_available = false;
    unavailable_reason->clear();

    if (!std::filesystem::exists(path)) {
        *unavailable_reason = "no_previous_run_baseline";
        return Status::Ok();
    }

    std::vector<std::uint8_t> bytes;
    const Status rd = codec::read_file_bytes(path, &bytes);
    if (!rd.ok) {
        *unavailable_reason = "baseline_read_failed";
        return Status::Ok();
    }
    const std::string body(bytes.begin(), bytes.end());
    if (body.empty()) {
        *unavailable_reason = "baseline_empty";
        return Status::Ok();
    }

    const std::regex row_re(
        "\\{\"stage_id\":\"([^\"]+)\",\"stage_elapsed_ms\":(-?[0-9]+(\\.[0-9]+)?)\\}");
    auto begin = std::sregex_iterator(body.begin(), body.end(), row_re);
    const auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        const std::smatch& m = *it;
        if (m.size() < 3) {
            continue;
        }
        try {
            const std::string id = m[1].str();
            const double elapsed_ms = std::stod(m[2].str());
            if (elapsed_ms >= 0.0 && std::isfinite(elapsed_ms)) {
                (*out)[id] = elapsed_ms;
            }
        } catch (...) {
            // Ignore malformed rows and continue parsing best-effort.
        }
    }
    if (out->empty()) {
        *unavailable_reason = "baseline_parse_failed";
        return Status::Ok();
    }
    *previous_run_available = true;
    return Status::Ok();
}

Status write_stage_baseline(
    const std::filesystem::path& path,
    const std::unordered_map<std::string, double>& stage_elapsed_ms) {
    std::vector<std::string> stage_ids;
    stage_ids.reserve(stage_elapsed_ms.size());
    for (const auto& kv : stage_elapsed_ms) {
        stage_ids.push_back(kv.first);
    }
    std::sort(stage_ids.begin(), stage_ids.end());

    std::ostringstream os;
    for (const auto& id : stage_ids) {
        const auto it = stage_elapsed_ms.find(id);
        if (it == stage_elapsed_ms.end()) {
            continue;
        }
        const double elapsed_ms = std::isfinite(it->second) && it->second >= 0.0 ? it->second : 0.0;
        os << "{\"stage_id\":\"" << json_escape(id) << "\",\"stage_elapsed_ms\":"
           << std::fixed << std::setprecision(3) << elapsed_ms << "}\n";
    }
    const std::string body = os.str();
    const std::vector<std::uint8_t> bytes(body.begin(), body.end());
    return codec::write_atomic_bytes(path, bytes);
}

void emit_event(
    std::ostream& out,
    const EventType event_type,
    const std::string& stage_id,
    const std::string& stage_name,
    const std::string& status,
    const std::string& start_ts,
    const std::optional<std::string>& end_ts,
    double elapsed_ms,
    double pipeline_elapsed_ms,
    const std::string& active_pipeline_state,
    const std::vector<std::pair<std::string, std::string>>& extra) {
    auto is_json_scalar_or_container = [](const std::string& v) -> bool {
        if (v.empty()) {
            return false;
        }
        const char c = v[0];
        if (c == '[' || c == '{' || c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            return true;
        }
        return v == "true" || v == "false" || v == "null";
    };
    std::ostringstream os;
    const std::string event_type_s = event_type_to_string(event_type);
    const double safe_elapsed_ms = (!std::isfinite(elapsed_ms) || elapsed_ms < 0.0) ? 0.0 : elapsed_ms;
    const double safe_pipeline_elapsed_ms =
        (!std::isfinite(pipeline_elapsed_ms) || pipeline_elapsed_ms < 0.0) ? 0.0 : pipeline_elapsed_ms;
    os << "{"
       << "\"event_type\":\"" << json_escape(event_type_s) << "\","
       << "\"stage_id\":\"" << json_escape(stage_id) << "\","
       << "\"stage_name\":\"" << json_escape(stage_name) << "\","
       << "\"status\":\"" << json_escape(status) << "\","
       << "\"start_ts\":\"" << json_escape(start_ts) << "\",";
    if (end_ts.has_value()) {
        os << "\"end_ts\":\"" << json_escape(*end_ts) << "\",";
    }
    os << "\"elapsed_ms\":" << std::fixed << std::setprecision(3) << safe_elapsed_ms << ","
       << "\"pipeline_elapsed_ms\":" << std::fixed << std::setprecision(3) << safe_pipeline_elapsed_ms << ","
       << "\"active_pipeline_state\":\"" << json_escape(active_pipeline_state) << "\"";
    for (const auto& kv : extra) {
        os << ",\"" << json_escape(kv.first) << "\":";
        if (is_json_scalar_or_container(kv.second)) {
            os << kv.second;
        } else {
            os << "\"" << json_escape(kv.second) << "\"";
        }
    }
    os << "}";
    out << os.str() << "\n";
}

}  // namespace vector_db_v3::telemetry
