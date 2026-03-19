#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <vector>

#include "vector_db_v3/codec/artifacts.hpp"
#include "vector_db_v3/codec/endian.hpp"
#include "vector_db_v3/codec/io.hpp"
#include "vector_db_v3/paths.hpp"
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

void set_env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

std::vector<float> make_vec(float base) {
    std::vector<float> v(vector_db_v3::kVectorDim, 0.0f);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = base + static_cast<float>((i % 19U) * 0.001f);
    }
    return v;
}

bool corrupt_fp16_unsorted_ids(const fs::path& fp16_path) {
    std::vector<std::uint8_t> bytes;
    if (!vector_db_v3::codec::read_file_bytes(fp16_path, &bytes).ok) {
        return false;
    }
    if (bytes.size() < vector_db_v3::codec::kEmbeddingShardHeaderSize + 2U * vector_db_v3::codec::kEmbeddingShardRecordSizeFp16) {
        return false;
    }
    const std::size_t base0 = vector_db_v3::codec::kEmbeddingShardHeaderSize;
    const std::size_t base1 = base0 + vector_db_v3::codec::kEmbeddingShardRecordSizeFp16;
    const std::uint64_t id0 = vector_db_v3::codec::load_le_u64(bytes.data() + base0);
    const std::uint64_t id1 = vector_db_v3::codec::load_le_u64(bytes.data() + base1);
    vector_db_v3::codec::store_le_u64(bytes.data() + base0, id1);
    vector_db_v3::codec::store_le_u64(bytes.data() + base1, id0);
    return vector_db_v3::codec::write_atomic_bytes(fp16_path, bytes).ok;
}

}  // namespace

int main() {
    using namespace vector_db_v3;
    bool ok = true;

    const fs::path root = fs::temp_directory_path() / "vectordb_v3_precision_shard_alignment_failures";
    std::error_code ec;
    fs::remove_all(root, ec);

    set_env("VECTOR_DB_V3_COMPLIANCE_PROFILE", "pass");
    set_env("VECTOR_DB_V3_INTERNAL_SHARD_MODE", "fp16");
    set_env("VECTOR_DB_V3_INTERNAL_SHARD_REPAIR", "regenerate");

    VectorStore store(root.string());
    ok &= expect(store.init().ok, "init");
    ok &= expect(store.open().ok, "open");
    ok &= expect(store.insert(100U, make_vec(0.10f)).ok, "insert 100");
    ok &= expect(store.insert(101U, make_vec(0.20f)).ok, "insert 101");
    ok &= expect(store.insert(102U, make_vec(0.30f)).ok, "insert 102");

    const fs::path fp16 = paths::embeddings_fp16_bin(root);
    ok &= expect(fs::exists(fp16), "fp16 shard should exist");
    ok &= expect(corrupt_fp16_unsorted_ids(fp16), "corrupt fp16 ids");

    set_env("VECTOR_DB_V3_INTERNAL_SHARD_REPAIR", "fail");
    const Status strict_fail = store.build_top_clusters(7U);
    ok &= expect(!strict_fail.ok, "strict repair=fail must fail on corrupt shard");

    set_env("VECTOR_DB_V3_INTERNAL_SHARD_REPAIR", "regenerate");
    const Status regen_ok = store.build_top_clusters(7U);
    ok &= expect(regen_ok.ok, "repair=regenerate must recover from corruption");

    std::error_code rm_ec;
    fs::remove(fp16, rm_ec);
    set_env("VECTOR_DB_V3_INTERNAL_SHARD_REPAIR", "fallback");
    const Status fallback_ok = store.build_top_clusters(8U);
    ok &= expect(fallback_ok.ok, "repair=fallback must succeed when fp16 shard missing");

    set_env("VECTOR_DB_V3_INTERNAL_SHARD_MODE", "");
    set_env("VECTOR_DB_V3_INTERNAL_SHARD_REPAIR", "");
    set_env("VECTOR_DB_V3_COMPLIANCE_PROFILE", "");

    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_precision_shard_alignment_failures_tests: PASS\n";
    return 0;
}
