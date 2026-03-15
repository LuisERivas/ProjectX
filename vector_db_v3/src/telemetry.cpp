#include "vector_db_v3/telemetry.hpp"

#include <chrono>
#include <cctype>
#include <iomanip>
#include <sstream>

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

void emit_event(
    std::ostream& out,
    const std::string& event_type,
    const std::string& stage_id,
    const std::string& stage_name,
    const std::string& status,
    double elapsed_ms,
    double pipeline_elapsed_ms,
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
    os << "{"
       << "\"event_type\":\"" << json_escape(event_type) << "\","
       << "\"stage_id\":\"" << json_escape(stage_id) << "\","
       << "\"stage_name\":\"" << json_escape(stage_name) << "\","
       << "\"status\":\"" << json_escape(status) << "\","
       << "\"start_ts\":\"" << now_ts() << "\","
       << "\"end_ts\":\"" << now_ts() << "\","
       << "\"elapsed_ms\":" << std::fixed << std::setprecision(3) << elapsed_ms << ","
       << "\"pipeline_elapsed_ms\":" << std::fixed << std::setprecision(3) << pipeline_elapsed_ms << ","
       << "\"active_pipeline_state\":\"scaffold\"";
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
