#include <cmath>
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

std::vector<float> make_sparse(std::initializer_list<std::pair<std::size_t, float>> values) {
    std::vector<float> v(vector_db_v3::kVectorDim, 0.0f);
    for (const auto& kv : values) {
        if (kv.first < v.size()) {
            v[kv.first] = kv.second;
        }
    }
    return v;
}

}  // namespace

int main() {
    const fs::path data_dir = fs::temp_directory_path() / "vectordb_v3_exact_search_test";
    std::error_code ec;
    fs::remove_all(data_dir, ec);

    bool ok = true;
    vector_db_v3::VectorStore store(data_dir.string());
    ok &= expect(store.init().ok, "init");
    ok &= expect(store.open().ok, "open");

    // Tie on score between ids 5 and 10; tie-break must prefer smaller id.
    ok &= expect(store.insert(10U, make_sparse({{0U, 1.0f}})).ok, "insert id=10");
    ok &= expect(store.insert(5U, make_sparse({{0U, 1.0f}})).ok, "insert id=5");
    ok &= expect(store.insert(20U, make_sparse({{0U, 0.5f}})).ok, "insert id=20");

    const auto q = make_sparse({{0U, 1.0f}});
    const auto all = store.search_exact(q, 10);
    ok &= expect(all.size() == 3U, "topk > live_rows should return all");
    if (all.size() == 3U) {
        ok &= expect(all[0].embedding_id == 5U, "tie-break first result must be smaller embedding_id");
        ok &= expect(all[1].embedding_id == 10U, "tie-break second result");
        ok &= expect(all[2].embedding_id == 20U, "lower score should rank later");
        ok &= expect(std::fabs(all[0].score - 1.0) < 1e-9, "id=5 score");
        ok &= expect(std::fabs(all[1].score - 1.0) < 1e-9, "id=10 score");
        ok &= expect(std::fabs(all[2].score - 0.5) < 1e-9, "id=20 score");
    }

    const auto top2 = store.search_exact(q, 2);
    ok &= expect(top2.size() == 2U, "topk truncation should apply");
    if (top2.size() == 2U) {
        ok &= expect(top2[0].embedding_id == 5U && top2[1].embedding_id == 10U, "top2 deterministic order");
    }

    const auto top0 = store.search_exact(q, 0);
    ok &= expect(top0.empty(), "topk=0 should return empty");

    const auto bad_dim = store.search_exact(std::vector<float>{1.0f, 2.0f}, 2);
    ok &= expect(bad_dim.empty(), "dimension mismatch should return empty");

    ok &= expect(store.close().ok, "close");

    vector_db_v3::VectorStore empty_store((data_dir.string() + "_empty"));
    ok &= expect(empty_store.init().ok, "init empty store");
    ok &= expect(empty_store.open().ok, "open empty store");
    const auto empty = empty_store.search_exact(q, 3);
    ok &= expect(empty.empty(), "empty store should return empty results");
    ok &= expect(empty_store.close().ok, "close empty store");

    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_exact_search_tests: PASS\n";
    return 0;
}
