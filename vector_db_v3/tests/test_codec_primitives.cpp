#include <filesystem>
#include <iostream>
#include <vector>

#include "vector_db_v3/codec/checksum.hpp"
#include "vector_db_v3/codec/endian.hpp"
#include "vector_db_v3/codec/io.hpp"

namespace fs = std::filesystem;

namespace {

bool expect(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    bool ok = true;

    std::vector<std::uint8_t> bytes(16, 0U);
    vector_db_v3::codec::store_le_u16(bytes.data() + 0, 0x1234U);
    vector_db_v3::codec::store_le_u32(bytes.data() + 2, 0xAABBCCDDU);
    vector_db_v3::codec::store_le_u64(bytes.data() + 6, 0x1122334455667788ULL);
    vector_db_v3::codec::store_le_f32(bytes.data() + 14 - 4, 1.5f);

    ok &= expect(vector_db_v3::codec::load_le_u16(bytes.data() + 0) == 0x1234U, "u16 roundtrip");
    ok &= expect(vector_db_v3::codec::load_le_u32(bytes.data() + 2) == 0xAABBCCDDU, "u32 roundtrip");
    ok &= expect(vector_db_v3::codec::load_le_u64(bytes.data() + 6) == 0x1122334455667788ULL, "u64 roundtrip");
    ok &= expect(vector_db_v3::codec::load_le_f32(bytes.data() + 10) == 1.5f, "f32 roundtrip");
    ok &= expect(vector_db_v3::codec::host_is_little_endian(), "host endianness expected little");

    const std::vector<std::uint8_t> crc_input = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
    ok &= expect(vector_db_v3::codec::crc32(crc_input) == 0xCBF43926U, "crc32 known vector");

    const std::vector<std::uint8_t> sha_input = {'a', 'b', 'c'};
    ok &= expect(
        vector_db_v3::codec::sha256_hex(sha_input) ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "sha256 known vector");

    const fs::path temp_root = fs::temp_directory_path() / "vectordb_v3_codec_primitives";
    std::error_code ec;
    fs::remove_all(temp_root, ec);
    fs::create_directories(temp_root, ec);
    const fs::path file = temp_root / "atomic.bin";

    const std::vector<std::uint8_t> one = {1U, 2U, 3U, 4U};
    const std::vector<std::uint8_t> two = {9U, 8U, 7U};

    ok &= expect(vector_db_v3::codec::write_atomic_bytes(file, one).ok, "atomic write first");
    ok &= expect(vector_db_v3::codec::write_atomic_bytes(file, two).ok, "atomic overwrite");
    std::vector<std::uint8_t> loaded;
    ok &= expect(vector_db_v3::codec::read_file_bytes(file, &loaded).ok, "atomic read");
    ok &= expect(loaded == two, "atomic overwrite visible");

    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_codec_primitives_tests: PASS\n";
    return 0;
}
