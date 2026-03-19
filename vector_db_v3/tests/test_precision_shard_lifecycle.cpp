#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <vector>

#include "vector_db_v3/codec/artifacts.hpp"
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
        v[i] = base + static_cast<float>((i % 17U) * 0.001f);
    }
    return v;
}

}  // namespace

int main() {
    using namespace vector_db_v3;
    bool ok = true;

    const fs::path root = fs::temp_directory_path() / "vectordb_v3_precision_shard_lifecycle";
    std::error_code ec;
    fs::remove_all(root, ec);

    set_env("VECTOR_DB_V3_COMPLIANCE_PROFILE", "pass");
    set_env("VECTOR_DB_V3_INTERNAL_SHARD_MODE", "fp16");
    set_env("VECTOR_DB_V3_INTERNAL_SHARD_REPAIR", "regenerate");

    VectorStore store(root.string());
    ok &= expect(store.init().ok, "init");
    ok &= expect(store.open().ok, "open");
    ok &= expect(store.insert(10U, make_vec(0.10f)).ok, "insert 10");
    ok &= expect(store.insert(20U, make_vec(0.20f)).ok, "insert 20");
    ok &= expect(store.insert(30U, make_vec(0.30f)).ok, "insert 30");

    const fs::path fp32 = paths::embeddings_fp32_bin(root);
    const fs::path fp16 = paths::embeddings_fp16_bin(root);
    ok &= expect(fs::exists(fp32), "embeddings_fp32.bin should exist");
    ok &= expect(fs::exists(fp16), "embeddings_fp16.bin should exist");

    std::vector<std::uint64_t> ids_fp32;
    std::vector<std::vector<float>> vecs_fp32;
    std::vector<std::uint64_t> ids_fp16;
    std::vector<std::vector<float>> vecs_fp16;
    ok &= expect(codec::read_embeddings_fp32_file(fp32, &ids_fp32, &vecs_fp32).ok, "read fp32 shard");
    ok &= expect(codec::read_embeddings_fp16_file(fp16, &ids_fp16, &vecs_fp16).ok, "read fp16 shard");
    ok &= expect(codec::validate_precision_id_alignment(ids_fp32, ids_fp16, {}).ok, "fp32/fp16 id alignment");

    set_env("VECTOR_DB_V3_INTERNAL_SHARD_MODE", "int8");
    ok &= expect(store.insert(40U, make_vec(0.40f)).ok, "insert 40 with int8 mode");
    const fs::path int8 = paths::embeddings_int8_sym_bin(root);
    ok &= expect(fs::exists(int8), "embeddings_int8_sym.bin should exist");

    std::vector<std::uint64_t> ids_int8;
    std::vector<std::vector<float>> vecs_int8;
    ok &= expect(codec::read_embeddings_int8_sym_file(int8, &ids_int8, &vecs_int8).ok, "read int8 shard");

    ids_fp32.clear();
    vecs_fp32.clear();
    ok &= expect(codec::read_embeddings_fp32_file(fp32, &ids_fp32, &vecs_fp32).ok, "re-read fp32 shard");
    ok &= expect(codec::validate_precision_id_alignment(ids_fp32, std::nullopt, {ids_int8}).ok, "fp32/int8 id alignment");

    set_env("VECTOR_DB_V3_INTERNAL_SHARD_MODE", "");
    set_env("VECTOR_DB_V3_INTERNAL_SHARD_REPAIR", "");
    set_env("VECTOR_DB_V3_COMPLIANCE_PROFILE", "");

    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_precision_shard_lifecycle_tests: PASS\n";
    return 0;
}
