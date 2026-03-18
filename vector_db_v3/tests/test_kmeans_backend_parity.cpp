#include <cmath>
#include <iostream>
#include <vector>

#include "vector_db_v3/kmeans_backend.hpp"

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
        v[i] = base + static_cast<float>((i % 17U) * 0.001f);
    }
    return v;
}

}  // namespace

int main() {
    using namespace vector_db_v3;

    std::vector<std::vector<float>> vectors;
    vectors.reserve(96U);
    for (std::size_t i = 0; i < 48U; ++i) {
        vectors.push_back(make_vec(0.1f));
    }
    for (std::size_t i = 0; i < 48U; ++i) {
        vectors.push_back(make_vec(0.9f));
    }

    bool ok = true;
    kmeans::KMeansResult cpu{};
    kmeans::KMeansResult candidate{};
    std::string backend;

    Status st = kmeans::run_kmeans(vectors, 8U, 8U, kmeans::BackendPreference::Cpu, &cpu, &backend);
    ok &= expect(st.ok, "cpu run should succeed");
    ok &= expect(backend == "cpu", "backend should be cpu");

    const bool cuda_available = kmeans::cuda_backend_available(nullptr);
    st = kmeans::run_kmeans(vectors, 8U, 8U, cuda_available ? kmeans::BackendPreference::Cuda : kmeans::BackendPreference::Auto, &candidate, &backend);
    ok &= expect(st.ok, "candidate run should succeed");

    ok &= expect(cpu.assignments.size() == candidate.assignments.size(), "assignment size parity");
    if (cpu.assignments.size() == candidate.assignments.size()) {
        for (std::size_t i = 0; i < cpu.assignments.size(); ++i) {
            if (cpu.assignments[i] != candidate.assignments[i]) {
                ok &= expect(false, "assignment parity mismatch");
                break;
            }
        }
    }

    const double delta = std::fabs(cpu.objective - candidate.objective);
    const double rel = cpu.objective > 0.0 ? delta / cpu.objective : delta;
    ok &= expect(rel <= 1e-4, "objective relative tolerance");

    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_kmeans_backend_parity_tests: PASS\n";
    return 0;
}
