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

std::vector<float> make_constant(float value) {
    return std::vector<float>(vector_db_v3::kVectorDim, value);
}

}  // namespace

int main() {
    using namespace vector_db_v3;
    bool ok = true;

    std::vector<std::vector<float>> vectors;
    for (std::size_t i = 0; i < 16U; ++i) {
        vectors.push_back(make_constant(1.0f));
    }

    kmeans::KMeansResult cpu{};
    std::string backend;
    Status st = kmeans::run_kmeans(vectors, 2U, 4U, kmeans::BackendPreference::Cpu, &cpu, &backend);
    ok &= expect(st.ok, "cpu kmeans should succeed");
    ok &= expect(cpu.assignments.size() == vectors.size(), "assignment size should match vectors");
    ok &= expect(cpu.assignments[0] == 1U, "empty-cluster repair should move first point to cluster 1");
    for (std::size_t i = 1; i < cpu.assignments.size(); ++i) {
        if (cpu.assignments[i] != 0U) {
            ok &= expect(false, "equal-distance ties should remain in earliest centroid");
            break;
        }
    }

    if (kmeans::cuda_backend_available(nullptr)) {
        kmeans::KMeansResult cuda{};
        st = kmeans::run_kmeans(vectors, 2U, 4U, kmeans::BackendPreference::Cuda, &cuda, &backend);
        ok &= expect(st.ok, "cuda kmeans should succeed when available");
        ok &= expect(cuda.assignments == cpu.assignments, "cuda assignments should match deterministic cpu baseline");
    }

    if (!ok) {
        return 1;
    }
    std::cout << "vectordb_v3_kmeans_tie_break_determinism_tests: PASS\n";
    return 0;
}
