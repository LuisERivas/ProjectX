#include <filesystem>
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
        v[i] = base + static_cast<float>(i) * 0.002f;
    }
    return v;
}

}  // namespace

int main() {
    const fs::path data_dir = fs::temp_directory_path() / "vectordb_v3_durability_checkpoint_test";
    std::error_code ec;
    fs::remove_all(data_dir, ec);

    bool ok = true;
    {
        vector_db_v3::VectorStore store(data_dir.string());
        ok &= expect(store.init().ok, "init");
        ok &= expect(store.open().ok, "open");
        ok &= expect(store.insert(1U, make_vec(1.0f)).ok, "insert1");
        ok &= expect(store.insert(2U, make_vec(2.0f)).ok, "insert2");
        ok &= expect(store.checkpoint().ok, "checkpoint");
        const auto ws = store.wal_stats();
        ok &= expect(ws.checkpoint_lsn == 2U, "checkpoint_lsn should equal 2");
        ok &= expect(ws.last_lsn == 2U, "last_lsn should remain 2");
        ok &= expect(ws.wal_entries == 0U, "wal entries reset after checkpoint");
        ok &= expect(store.close().ok, "close");
    }

    {
        vector_db_v3::VectorStore store(data_dir.string());
        ok &= expect(store.open().ok, "reopen");
        ok &= expect(store.get(1U).has_value(), "record1 restored");
        ok &= expect(store.get(2U).has_value(), "record2 restored");
        ok &= expect(store.insert(3U, make_vec(3.0f)).ok, "insert3 post-checkpoint");
        const auto ws = store.wal_stats();
        ok &= expect(ws.last_lsn == 3U, "lsn increments from checkpoint");
        ok &= expect(ws.wal_entries == 1U, "wal entries after new insert");
        ok &= expect(store.close().ok, "close2");
    }

    {
        vector_db_v3::VectorStore store(data_dir.string());
        ok &= expect(store.open().ok, "reopen2");
        ok &= expect(store.get(3U).has_value(), "record3 replayed");
        ok &= expect(store.close().ok, "close3");
    }

    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_durability_checkpoint_tests: PASS\n";
    return 0;
}
