#include <filesystem>
#include <iostream>

#include "vector_db_v3/vector_store.hpp"

namespace fs = std::filesystem;

int main() {
    const fs::path data_dir = fs::temp_directory_path() / "vectordb_v3_scaffold_test_data";
    std::error_code ec;
    fs::remove_all(data_dir, ec);

    vector_db_v3::VectorStore store(data_dir.string());
    auto s = store.init();
    if (!s.ok) {
        std::cerr << "init failed: " << s.message << "\n";
        return 1;
    }
    if (!fs::exists(data_dir / "manifest.json") ||
        !fs::exists(data_dir / "wal.log") ||
        !fs::exists(data_dir / "segments") ||
        !fs::exists(data_dir / "clusters" / "current")) {
        std::cerr << "bootstrap layout missing\n";
        return 1;
    }

    s = store.open();
    if (!s.ok) {
        std::cerr << "open failed: " << s.message << "\n";
        return 1;
    }

    const auto st = store.stats();
    if (st.dimension != vector_db_v3::kVectorDim || st.total_rows != 0 || st.live_rows != 0 || st.tombstone_rows != 0) {
        std::cerr << "stats mismatch\n";
        return 1;
    }

    s = store.build_top_clusters(1234);
    if (!s.ok) {
        std::cerr << "top stage stub failed: " << s.message << "\n";
        return 1;
    }

    s = store.insert(1, std::vector<float>(vector_db_v3::kVectorDim, 0.0f));
    if (!s.ok) {
        std::cerr << "insert should succeed in durability section: " << s.message << "\n";
        return 1;
    }
    const auto rec = store.get(1);
    if (!rec.has_value() || rec->vector.size() != vector_db_v3::kVectorDim) {
        std::cerr << "insert/get durability path failed\n";
        return 1;
    }

    s = store.close();
    if (!s.ok) {
        std::cerr << "close failed: " << s.message << "\n";
        return 1;
    }

    std::cout << "vectordb_v3_scaffold_tests: PASS\n";
    return 0;
}
