#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "vector_db/vector_store.hpp"

namespace fs = std::filesystem;

namespace {

bool expect(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << "\n";
        return false;
    }
    return true;
}

std::vector<float> make_vec(float base) {
    std::vector<float> v(vector_db::kVectorDim, 0.0f);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = base + static_cast<float>(i) * 0.001f;
    }
    return v;
}

}  // namespace

int main() {
    const fs::path test_dir = fs::path("vector_db_test_data");
    std::error_code ec;
    fs::remove_all(test_dir, ec);

    vector_db::VectorStore store(test_dir.string());
    if (!expect(store.init().ok, "init succeeds")) {
        return 1;
    }
    if (!expect(store.open().ok, "open succeeds")) {
        return 1;
    }

    const auto v1 = make_vec(1.0f);
    const auto v2 = make_vec(2.0f);

    if (!expect(store.insert(42, v1, "{\"tag\":\"first\"}").ok, "insert #1")) {
        return 1;
    }
    if (!expect(store.insert(7, v2, "{\"tag\":\"second\"}").ok, "insert #2")) {
        return 1;
    }

    if (!expect(!store.insert(42, v1, "{\"tag\":\"dup\"}").ok, "duplicate id rejected")) {
        return 1;
    }

    const auto rec42 = store.get(42);
    if (!expect(rec42.has_value(), "get existing id")) {
        return 1;
    }
    if (!expect(!rec42->deleted, "record not deleted initially")) {
        return 1;
    }
    if (!expect(rec42->vector_fp32.size() == vector_db::kVectorDim, "vector has 1024 dims")) {
        return 1;
    }

    if (!expect(store.update_metadata(42, "{\"tag\":\"updated\",\"k\":\"v\"}").ok, "metadata update")) {
        return 1;
    }
    const auto rec42b = store.get(42);
    if (!expect(rec42b.has_value() && rec42b->metadata_json.find("updated") != std::string::npos, "metadata persisted in memory")) {
        return 1;
    }

    if (!expect(store.remove(42).ok, "delete existing id")) {
        return 1;
    }
    const auto rec42c = store.get(42);
    if (!expect(rec42c.has_value() && rec42c->deleted, "tombstone behavior")) {
        return 1;
    }

    const auto st = store.stats();
    if (!expect(st.total_rows == 2, "stats total rows")) {
        return 1;
    }
    if (!expect(st.live_rows == 1, "stats live rows")) {
        return 1;
    }
    if (!expect(st.tombstone_rows == 1, "stats tombstone rows")) {
        return 1;
    }
    if (!expect(st.dirty_ranges >= 4, "dirty ranges tracked")) {
        return 1;
    }
    if (!expect(fs::exists(test_dir / "manifest.json"), "manifest file written")) {
        return 1;
    }
    if (!expect(fs::exists(test_dir / "dirty_ranges.json"), "dirty ranges file written")) {
        return 1;
    }

    if (!expect(store.close().ok, "close succeeds")) {
        return 1;
    }

    vector_db::VectorStore reopened(test_dir.string());
    if (!expect(reopened.open().ok, "reopen succeeds")) {
        return 1;
    }
    const auto rec7 = reopened.get(7);
    if (!expect(rec7.has_value() && !rec7->deleted, "reloaded live record")) {
        return 1;
    }
    const auto rec42d = reopened.get(42);
    if (!expect(rec42d.has_value() && rec42d->deleted, "reloaded tombstoned record")) {
        return 1;
    }
    const auto st_reopened = reopened.stats();
    if (!expect(st_reopened.total_rows == st.total_rows, "manifest reload keeps row counts")) {
        return 1;
    }
    if (!expect(st_reopened.tombstone_rows == st.tombstone_rows, "manifest/tombstone reload keeps tombstones")) {
        return 1;
    }
    if (!expect(st_reopened.dirty_ranges == st.dirty_ranges, "dirty ranges reload completion")) {
        return 1;
    }

    std::cout << "[PASS] vectordb phase1 tests\n";
    reopened.close();
    fs::remove_all(test_dir, ec);
    return 0;
}

