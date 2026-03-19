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
        v[i] = base + static_cast<float>(i) * 0.001f;
    }
    return v;
}

}  // namespace

int main() {
    const fs::path data_dir = fs::temp_directory_path() / "vectordb_v3_durability_wal_test";
    std::error_code ec;
    fs::remove_all(data_dir, ec);

    bool ok = true;

    {
        vector_db_v3::VectorStore store(data_dir.string());
        ok &= expect(store.init().ok, "init");
        ok &= expect(store.open().ok, "open");
        ok &= expect(store.insert(10U, make_vec(1.0f)).ok, "insert 10");
        ok &= expect(store.insert(20U, make_vec(2.0f)).ok, "insert 20");
        std::vector<vector_db_v3::Record> batch = {
            vector_db_v3::Record{30U, make_vec(3.0f)},
            vector_db_v3::Record{40U, make_vec(4.0f)},
        };
        ok &= expect(store.insert_batch(batch).ok, "insert_batch 30/40");
        const auto ws = store.wal_stats();
        ok &= expect(ws.last_lsn == 4U, "lsn should be 4");
        ok &= expect(ws.wal_entries == 4U, "wal entries should be 4");
        ok &= expect(store.close().ok, "close");
    }

    {
        vector_db_v3::VectorStore store(data_dir.string());
        ok &= expect(store.open().ok, "reopen");
        auto rec = store.get(10U);
        ok &= expect(rec.has_value(), "record 10 exists after replay");
        ok &= expect(store.get(30U).has_value(), "record 30 exists after replay");
        ok &= expect(store.get(40U).has_value(), "record 40 exists after replay");
        ok &= expect(store.remove(10U).ok, "delete 10");
        ok &= expect(store.get(10U).has_value() == false, "record 10 deleted in memory");
        ok &= expect(store.close().ok, "close2");
    }

    {
        vector_db_v3::VectorStore store(data_dir.string());
        ok &= expect(store.open().ok, "reopen2");
        ok &= expect(store.get(10U).has_value() == false, "record 10 deleted after replay");
        ok &= expect(store.get(20U).has_value(), "record 20 still exists");
        ok &= expect(store.close().ok, "close3");
    }

    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_durability_wal_tests: PASS\n";
    return 0;
}
