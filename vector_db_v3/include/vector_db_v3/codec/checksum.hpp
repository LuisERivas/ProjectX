#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vector_db_v3::codec {

std::uint32_t crc32(const std::uint8_t* data, std::size_t size) noexcept;
std::uint32_t crc32(const std::vector<std::uint8_t>& data) noexcept;
std::string sha256_hex(const std::vector<std::uint8_t>& data);

}  // namespace vector_db_v3::codec
