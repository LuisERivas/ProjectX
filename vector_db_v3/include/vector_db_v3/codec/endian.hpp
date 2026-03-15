#pragma once

#include <cstdint>
#include <cstring>

namespace vector_db_v3::codec {

inline bool host_is_little_endian() noexcept {
    const std::uint16_t v = 0x0102;
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    return p[0] == 0x02;
}

inline std::uint16_t load_le_u16(const std::uint8_t* p) noexcept {
    return static_cast<std::uint16_t>(p[0]) |
           (static_cast<std::uint16_t>(p[1]) << 8U);
}

inline std::uint32_t load_le_u32(const std::uint8_t* p) noexcept {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8U) |
           (static_cast<std::uint32_t>(p[2]) << 16U) |
           (static_cast<std::uint32_t>(p[3]) << 24U);
}

inline std::uint64_t load_le_u64(const std::uint8_t* p) noexcept {
    return static_cast<std::uint64_t>(p[0]) |
           (static_cast<std::uint64_t>(p[1]) << 8U) |
           (static_cast<std::uint64_t>(p[2]) << 16U) |
           (static_cast<std::uint64_t>(p[3]) << 24U) |
           (static_cast<std::uint64_t>(p[4]) << 32U) |
           (static_cast<std::uint64_t>(p[5]) << 40U) |
           (static_cast<std::uint64_t>(p[6]) << 48U) |
           (static_cast<std::uint64_t>(p[7]) << 56U);
}

inline float load_le_f32(const std::uint8_t* p) noexcept {
    const std::uint32_t bits = load_le_u32(p);
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

inline void store_le_u16(std::uint8_t* p, std::uint16_t v) noexcept {
    p[0] = static_cast<std::uint8_t>(v & 0xFFU);
    p[1] = static_cast<std::uint8_t>((v >> 8U) & 0xFFU);
}

inline void store_le_u32(std::uint8_t* p, std::uint32_t v) noexcept {
    p[0] = static_cast<std::uint8_t>(v & 0xFFU);
    p[1] = static_cast<std::uint8_t>((v >> 8U) & 0xFFU);
    p[2] = static_cast<std::uint8_t>((v >> 16U) & 0xFFU);
    p[3] = static_cast<std::uint8_t>((v >> 24U) & 0xFFU);
}

inline void store_le_u64(std::uint8_t* p, std::uint64_t v) noexcept {
    p[0] = static_cast<std::uint8_t>(v & 0xFFU);
    p[1] = static_cast<std::uint8_t>((v >> 8U) & 0xFFU);
    p[2] = static_cast<std::uint8_t>((v >> 16U) & 0xFFU);
    p[3] = static_cast<std::uint8_t>((v >> 24U) & 0xFFU);
    p[4] = static_cast<std::uint8_t>((v >> 32U) & 0xFFU);
    p[5] = static_cast<std::uint8_t>((v >> 40U) & 0xFFU);
    p[6] = static_cast<std::uint8_t>((v >> 48U) & 0xFFU);
    p[7] = static_cast<std::uint8_t>((v >> 56U) & 0xFFU);
}

inline void store_le_f32(std::uint8_t* p, float v) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(float));
    store_le_u32(p, bits);
}

}  // namespace vector_db_v3::codec
