#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "vector_db_v3/status.hpp"

namespace vector_db_v3::codec {

Status read_file_bytes(const std::filesystem::path& path, std::vector<std::uint8_t>* out);
Status write_atomic_bytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes);
Status validate_exact_size_multiple(
    std::size_t bytes,
    std::size_t record_size,
    const std::string& artifact_name);
std::filesystem::path temp_path_for(const std::filesystem::path& target);

}  // namespace vector_db_v3::codec
