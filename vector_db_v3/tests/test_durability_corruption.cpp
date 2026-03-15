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
        v[i] = base + static_cast<float>(i) * 0.004f;
    }
    return v;
}

bool corrupt_middle_byte(const fs::path& p) {
    std::fstream io(p, std::ios::binary | std::ios::in | std::ios::out);
    if (!io) {
        return false;
    }
    io.seekg(0, std::ios::end);
    const std::streamoff sz = io.tellg();
    if (sz < 64) {
        return false;
    }
    const std::streamoff pos = sz / 2;
    io.seekg(pos, std::ios::beg);
    char c = 0;
    io.read(&c, 1);
    if (!io) {
        return false;
    }
    c = static_cast<char>(c ^ 0x7F);
    io.seekp(pos, std::ios::beg);
    io.write(&c, 1);
    return static_cast<bool>(io);
}

}  // namespace

int main() {
    const fs::path data_dir = fs::temp_directory_path() / "vectordb_v3_durability_corruption_test";
    std::error_code ec;
    fs::remove_all(data_dir, ec);

    bool ok = true;
    {
        vector_db_v3::VectorStore store(data_dir.string());
        ok &= expect(store.init().ok, "init");
        ok &= expect(store.open().ok, "open");
        ok &= expect(store.insert(42U, make_vec(1.0f)).ok, "insert");
        ok &= expect(store.close().ok, "close");
    }

    const fs::path wal = data_dir / "wal.log";
    ok &= expect(corrupt_middle_byte(wal), "corrupt wal");

    {
        vector_db_v3::VectorStore store(data_dir.string());
        const auto s = store.open();
        ok &= expect(!s.ok, "open must fail on unrecoverable wal corruption");
    }

    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_durability_corruption_tests: PASS\n";
    return 0;
}
