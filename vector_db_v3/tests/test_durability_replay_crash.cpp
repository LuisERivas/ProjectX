#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "vector_db_v3/vector_store.hpp"

namespace fs = std::filesystem;

namespace {

bool expect(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        return false;
    }
    return true;
}

std::vector<float> make_vec(float base) {
    std::vector<float> v(vector_db_v3::kVectorDim, 0.0f);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = base + static_cast<float>(i) * 0.003f;
    }
    return v;
}

bool truncate_last_byte(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        return false;
    }
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.empty()) {
        return false;
    }
    bytes.pop_back();
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    if (!bytes.empty()) {
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    return static_cast<bool>(out);
}

}  // namespace

int main() {
    const fs::path data_dir = fs::temp_directory_path() / "vectordb_v3_durability_crash_test";
    std::error_code ec;
    fs::remove_all(data_dir, ec);

    bool ok = true;
    {
        vector_db_v3::VectorStore store(data_dir.string());
        ok &= expect(store.init().ok, "init");
        ok &= expect(store.open().ok, "open");
        ok &= expect(store.insert(100U, make_vec(1.0f)).ok, "insert first");
        ok &= expect(store.insert(200U, make_vec(2.0f)).ok, "insert second");
        ok &= expect(store.close().ok, "close");
    }

    const fs::path wal = data_dir / "wal.log";
    ok &= expect(truncate_last_byte(wal), "truncate wal tail");

    {
        vector_db_v3::VectorStore store(data_dir.string());
        ok &= expect(store.open().ok, "open after torn wal tail");
        ok &= expect(store.get(100U).has_value(), "first record survives");
        ok &= expect(!store.get(200U).has_value(), "second record removed due tail truncation");
        ok &= expect(store.close().ok, "close2");
    }

    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_durability_replay_crash_tests: PASS\n";
    return 0;
}
