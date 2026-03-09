#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
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

std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream os;
    os << in.rdbuf();
    return os.str();
}

std::size_t count_non_empty_lines(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::string line;
    std::size_t count = 0;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            count += 1;
        }
    }
    return count;
}

std::vector<std::uint64_t> parse_lsns(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::vector<std::uint64_t> out;
    std::string line;
    while (std::getline(in, line)) {
        const auto key = line.find("\"lsn\":");
        if (key == std::string::npos) {
            continue;
        }
        const auto n0 = line.find_first_of("0123456789", key);
        const auto n1 = line.find_first_not_of("0123456789", n0);
        out.push_back(static_cast<std::uint64_t>(std::stoull(line.substr(n0, n1 - n0))));
    }
    return out;
}

}  // namespace

int main() {
    const fs::path test_dir = fs::path("vector_db_test_data");
    std::error_code ec;
    fs::remove_all(test_dir, ec);

    const auto v1 = make_vec(1.0f);
    const auto v2 = make_vec(2.0f);

    // A) WAL append correctness + baseline CRUD path.
    vector_db::VectorStore store(test_dir.string());
    if (!expect(store.init().ok, "init succeeds")) {
        return 1;
    }
    if (!expect(store.open().ok, "open succeeds")) {
        return 1;
    }
    if (!expect(store.insert(42, v1, "{\"tag\":\"first\"}").ok, "insert #1")) {
        return 1;
    }
    if (!expect(store.update_metadata(42, "{\"tag\":\"updated\",\"k\":\"v\"}").ok, "metadata update")) {
        return 1;
    }
    if (!expect(store.remove(42).ok, "delete existing id")) {
        return 1;
    }

    const fs::path wal = test_dir / "wal.log";
    if (!expect(fs::exists(wal), "wal exists")) {
        return 1;
    }
    if (!expect(count_non_empty_lines(wal) == 3, "wal has 3 records")) {
        return 1;
    }
    const auto wal_text = slurp(wal);
    if (!expect(wal_text.find("\"op\":\"INSERT\"") != std::string::npos, "wal contains INSERT")) {
        return 1;
    }
    if (!expect(wal_text.find("\"op\":\"UPDATE_META\"") != std::string::npos, "wal contains UPDATE_META")) {
        return 1;
    }
    if (!expect(wal_text.find("\"op\":\"DELETE\"") != std::string::npos, "wal contains DELETE")) {
        return 1;
    }
    const auto lsns = parse_lsns(wal);
    if (!expect(lsns.size() == 3, "lsn parsed for every wal record")) {
        return 1;
    }
    if (!expect(lsns[0] < lsns[1] && lsns[1] < lsns[2], "lsn monotonic increase")) {
        return 1;
    }

    // Baseline state checks.
    const auto rec42 = store.get(42);
    if (!expect(rec42.has_value() && rec42->deleted, "record tombstoned after delete")) {
        return 1;
    }
    if (!expect(store.insert(7, v2, "{\"tag\":\"second\"}").ok, "insert #2")) {
        return 1;
    }
    if (!expect(!store.insert(7, v2, "{\"tag\":\"dup\"}").ok, "duplicate id rejected")) {
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
    if (!expect(fs::exists(test_dir / "manifest.json"), "manifest file written")) {
        return 1;
    }
    if (!expect(fs::exists(test_dir / "dirty_ranges.json"), "dirty ranges file written")) {
        return 1;
    }

    // B) Replay after simulated crash: no close, then reopen a fresh instance.
    vector_db::VectorStore crash_reopen(test_dir.string());
    if (!expect(crash_reopen.open().ok, "reopen after simulated crash succeeds")) {
        return 1;
    }
    const auto rec7 = crash_reopen.get(7);
    if (!expect(rec7.has_value() && !rec7->deleted, "reloaded live record after replay")) {
        return 1;
    }
    const auto rec42_after = crash_reopen.get(42);
    if (!expect(rec42_after.has_value() && rec42_after->deleted, "reloaded tombstoned record after replay")) {
        return 1;
    }

    // C) Idempotent repeated replay: second open should not change counts.
    const auto st_once = crash_reopen.stats();
    vector_db::VectorStore second_reopen(test_dir.string());
    if (!expect(second_reopen.open().ok, "second reopen succeeds")) {
        return 1;
    }
    const auto st_twice = second_reopen.stats();
    if (!expect(st_once.total_rows == st_twice.total_rows, "repeated open keeps total rows")) {
        return 1;
    }
    if (!expect(st_once.tombstone_rows == st_twice.tombstone_rows, "repeated open keeps tombstones")) {
        return 1;
    }

    // D) Checkpoint truncation behavior.
    if (!expect(second_reopen.checkpoint().ok, "checkpoint succeeds")) {
        return 1;
    }
    const auto wal_after_checkpoint = second_reopen.wal_stats();
    if (!expect(wal_after_checkpoint.wal_entries == 0, "wal truncated after checkpoint")) {
        return 1;
    }
    if (!expect(wal_after_checkpoint.checkpoint_lsn >= lsns.back(), "checkpoint_lsn advanced")) {
        return 1;
    }

    // E) Corrupted trailing WAL line tolerance.
    if (!expect(second_reopen.insert(99, v1, "{\"tag\":\"tail\"}").ok, "insert before corrupt tail")) {
        return 1;
    }
    {
        std::ofstream out(test_dir / "wal.log", std::ios::binary | std::ios::app);
        out << "{\"lsn\":999999,\"op\":\"INSERT\"";  // malformed trailing line
    }
    vector_db::VectorStore malformed_tail_reopen(test_dir.string());
    if (!expect(malformed_tail_reopen.open().ok, "open succeeds with malformed trailing wal line")) {
        return 1;
    }
    const auto rec99 = malformed_tail_reopen.get(99);
    if (!expect(rec99.has_value() && !rec99->deleted, "valid wal records still replay with malformed tail")) {
        return 1;
    }

    std::cout << "[PASS] vectordb phase2 wal/recovery tests\n";
    malformed_tail_reopen.close();
    fs::remove_all(test_dir, ec);
    return 0;
}

