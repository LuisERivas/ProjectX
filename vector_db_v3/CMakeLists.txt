cmake_minimum_required(VERSION 3.16)
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()
project(vectordb_v3_m1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()
endif()
find_package(CUDAToolkit QUIET)

set(VECTOR_DB_V3_ENABLE_CUDA_BACKEND OFF)
set(VECTOR_DB_V3_CUDART_TARGET "")
set(VECTOR_DB_V3_CUBLAS_TARGET "")
set(VECTOR_DB_V3_ENABLE_TENSOR_BACKEND OFF)
if(CMAKE_CUDA_COMPILER)
    if(TARGET CUDAToolkit::cudart)
        set(VECTOR_DB_V3_ENABLE_CUDA_BACKEND ON)
        set(VECTOR_DB_V3_CUDART_TARGET CUDAToolkit::cudart)
    elseif(TARGET CUDA::cudart)
        set(VECTOR_DB_V3_ENABLE_CUDA_BACKEND ON)
        set(VECTOR_DB_V3_CUDART_TARGET CUDA::cudart)
    endif()
    if(TARGET CUDAToolkit::cublas)
        set(VECTOR_DB_V3_CUBLAS_TARGET CUDAToolkit::cublas)
        set(VECTOR_DB_V3_ENABLE_TENSOR_BACKEND ON)
    elseif(TARGET CUDA::cublas)
        set(VECTOR_DB_V3_CUBLAS_TARGET CUDA::cublas)
        set(VECTOR_DB_V3_ENABLE_TENSOR_BACKEND ON)
    endif()
endif()

set(VECTOR_DB_V3_CORE_SOURCES
    src/vector_store.cpp
    src/telemetry.cpp
    src/kmeans_backend_cpu.cpp
    src/cuda/kmeans_runtime.cpp
    src/cuda/pipeline_context.cpp
    src/codec/checksum.cpp
    src/codec/io.cpp
    src/codec/artifacts.cpp
)

if(VECTOR_DB_V3_ENABLE_CUDA_BACKEND)
    list(APPEND VECTOR_DB_V3_CORE_SOURCES src/cuda/kmeans_kernels.cu src/cuda/tensor_runtime.cpp)
    if(VECTOR_DB_V3_ENABLE_TENSOR_BACKEND)
        list(APPEND VECTOR_DB_V3_CORE_SOURCES src/cuda/tensor_distance.cu)
    endif()
endif()

add_library(vectordb_v3 ${VECTOR_DB_V3_CORE_SOURCES})
target_include_directories(vectordb_v3 PUBLIC include)

if (MSVC)
    target_compile_options(vectordb_v3 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(vectordb_v3 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()

if(VECTOR_DB_V3_ENABLE_CUDA_BACKEND)
    target_compile_definitions(vectordb_v3 PUBLIC VECTOR_DB_V3_CUDA_ENABLED=1)
    target_link_libraries(vectordb_v3 PRIVATE ${VECTOR_DB_V3_CUDART_TARGET})
    if(VECTOR_DB_V3_ENABLE_TENSOR_BACKEND)
        target_compile_definitions(vectordb_v3 PUBLIC VECTOR_DB_V3_TENSOR_ENABLED=1)
        target_link_libraries(vectordb_v3 PRIVATE ${VECTOR_DB_V3_CUBLAS_TARGET})
    else()
        target_compile_definitions(vectordb_v3 PUBLIC VECTOR_DB_V3_TENSOR_ENABLED=0)
    endif()
    set_target_properties(vectordb_v3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    target_compile_definitions(vectordb_v3 PUBLIC VECTOR_DB_V3_CUDA_ENABLED=0)
    target_compile_definitions(vectordb_v3 PUBLIC VECTOR_DB_V3_TENSOR_ENABLED=0)
    if(CMAKE_CUDA_COMPILER)
        message(WARNING "CUDA compiler detected but cudart imported target not found; building CPU fallback only.")
    endif()
endif()

add_executable(vectordb_v3_cli
    cli/main.cpp
)
target_link_libraries(vectordb_v3_cli PRIVATE vectordb_v3)

add_executable(vectordb_v3_tests
    tests/test_scaffold.cpp
)
target_link_libraries(vectordb_v3_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_codec_primitives_tests
    tests/test_codec_primitives.cpp
)
target_link_libraries(vectordb_v3_codec_primitives_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_codec_artifacts_tests
    tests/test_codec_artifacts.cpp
)
target_link_libraries(vectordb_v3_codec_artifacts_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_codec_corruption_tests
    tests/test_codec_corruption.cpp
)
target_link_libraries(vectordb_v3_codec_corruption_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_durability_wal_tests
    tests/test_durability_wal.cpp
)
target_link_libraries(vectordb_v3_durability_wal_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_durability_checkpoint_tests
    tests/test_durability_checkpoint.cpp
)
target_link_libraries(vectordb_v3_durability_checkpoint_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_durability_replay_crash_tests
    tests/test_durability_replay_crash.cpp
)
target_link_libraries(vectordb_v3_durability_replay_crash_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_durability_corruption_tests
    tests/test_durability_corruption.cpp
)
target_link_libraries(vectordb_v3_durability_corruption_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_exact_search_tests
    tests/test_exact_search.cpp
)
target_link_libraries(vectordb_v3_exact_search_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_top_layer_artifacts_tests
    tests/test_top_layer_artifacts.cpp
)
target_link_libraries(vectordb_v3_top_layer_artifacts_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_mid_layer_artifacts_tests
    tests/test_mid_layer_artifacts.cpp
)
target_link_libraries(vectordb_v3_mid_layer_artifacts_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_lower_layer_gate_artifacts_tests
    tests/test_lower_layer_gate_artifacts.cpp
)
target_link_libraries(vectordb_v3_lower_layer_gate_artifacts_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_final_layer_artifacts_tests
    tests/test_final_layer_artifacts.cpp
)
target_link_libraries(vectordb_v3_final_layer_artifacts_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_final_layer_eligibility_reconciliation_tests
    tests/test_final_layer_eligibility_reconciliation.cpp
)
target_link_libraries(vectordb_v3_final_layer_eligibility_reconciliation_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_compliance_pass_tests
    tests/test_compliance_pass.cpp
)
target_link_libraries(vectordb_v3_compliance_pass_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_compliance_fail_fast_tests
    tests/test_compliance_fail_fast.cpp
)
target_link_libraries(vectordb_v3_compliance_fail_fast_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_kmeans_backend_parity_tests
    tests/test_kmeans_backend_parity.cpp
)
target_link_libraries(vectordb_v3_kmeans_backend_parity_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_kmeans_tie_break_determinism_tests
    tests/test_kmeans_tie_break_determinism.cpp
)
target_link_libraries(vectordb_v3_kmeans_tie_break_determinism_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_kmeans_backend_selection_tests
    tests/test_kmeans_backend_selection.cpp
)
target_link_libraries(vectordb_v3_kmeans_backend_selection_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_gpu_residency_tests
    tests/test_gpu_residency.cpp
)
target_link_libraries(vectordb_v3_gpu_residency_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_precision_shard_lifecycle_tests
    tests/test_precision_shard_lifecycle.cpp
)
target_link_libraries(vectordb_v3_precision_shard_lifecycle_tests PRIVATE vectordb_v3)

add_executable(vectordb_v3_precision_shard_alignment_failures_tests
    tests/test_precision_shard_alignment_failures.cpp
)
target_link_libraries(vectordb_v3_precision_shard_alignment_failures_tests PRIVATE vectordb_v3)

enable_testing()
add_test(NAME vectordb_v3_tests COMMAND vectordb_v3_tests)
add_test(NAME vectordb_v3_codec_primitives_tests COMMAND vectordb_v3_codec_primitives_tests)
add_test(NAME vectordb_v3_codec_artifacts_tests COMMAND vectordb_v3_codec_artifacts_tests)
add_test(NAME vectordb_v3_codec_corruption_tests COMMAND vectordb_v3_codec_corruption_tests)
add_test(NAME vectordb_v3_durability_wal_tests COMMAND vectordb_v3_durability_wal_tests)
add_test(NAME vectordb_v3_durability_checkpoint_tests COMMAND vectordb_v3_durability_checkpoint_tests)
add_test(NAME vectordb_v3_durability_replay_crash_tests COMMAND vectordb_v3_durability_replay_crash_tests)
add_test(NAME vectordb_v3_durability_corruption_tests COMMAND vectordb_v3_durability_corruption_tests)
add_test(NAME vectordb_v3_exact_search_tests COMMAND vectordb_v3_exact_search_tests)
add_test(NAME vectordb_v3_top_layer_artifacts_tests COMMAND vectordb_v3_top_layer_artifacts_tests)
add_test(NAME vectordb_v3_mid_layer_artifacts_tests COMMAND vectordb_v3_mid_layer_artifacts_tests)
add_test(NAME vectordb_v3_lower_layer_gate_artifacts_tests COMMAND vectordb_v3_lower_layer_gate_artifacts_tests)
add_test(NAME vectordb_v3_final_layer_artifacts_tests COMMAND vectordb_v3_final_layer_artifacts_tests)
add_test(
    NAME vectordb_v3_final_layer_eligibility_reconciliation_tests
    COMMAND vectordb_v3_final_layer_eligibility_reconciliation_tests)
add_test(NAME vectordb_v3_compliance_pass_tests COMMAND vectordb_v3_compliance_pass_tests)
add_test(NAME vectordb_v3_compliance_fail_fast_tests COMMAND vectordb_v3_compliance_fail_fast_tests)
add_test(NAME vectordb_v3_kmeans_backend_parity_tests COMMAND vectordb_v3_kmeans_backend_parity_tests)
add_test(
    NAME vectordb_v3_kmeans_tie_break_determinism_tests
    COMMAND vectordb_v3_kmeans_tie_break_determinism_tests)
add_test(NAME vectordb_v3_kmeans_backend_selection_tests COMMAND vectordb_v3_kmeans_backend_selection_tests)
add_test(NAME vectordb_v3_gpu_residency_tests COMMAND vectordb_v3_gpu_residency_tests)
add_test(NAME vectordb_v3_precision_shard_lifecycle_tests COMMAND vectordb_v3_precision_shard_lifecycle_tests)
add_test(
    NAME vectordb_v3_precision_shard_alignment_failures_tests
    COMMAND vectordb_v3_precision_shard_alignment_failures_tests)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME vectordb_v3_cli_contract_tests
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_cli_contract.py")
    add_test(
        NAME vectordb_v3_terminal_event_contract_tests
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_terminal_event_contract.py")
    add_test(
        NAME vectordb_v3_smoke_scaffold_tests
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/smoke_scaffold.py")
    add_test(
        NAME vectordb_v3_traceability_verify_tests
        COMMAND
            ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_traceability.py"
            --matrix "${CMAKE_CURRENT_SOURCE_DIR}/contracts/TRACEABILITY_MATRIX.md"
            --gate-map "${CMAKE_CURRENT_SOURCE_DIR}/scripts/gate_map_reference.json"
            --out-dir "${CMAKE_CURRENT_BINARY_DIR}/gate_evidence/traceability_ctest")
    add_test(
        NAME vectordb_v3_perf_gate_soft_tests
        COMMAND
            ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/perf_gate.py"
            --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
            --mode evidence
            --profile minimum
            --out-dir "${CMAKE_CURRENT_BINARY_DIR}/gate_evidence/perf_ctest")
    add_test(
        NAME vectordb_v3_perf_gate_baseline_tests
        COMMAND
            ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/perf_gate.py"
            --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
            --mode baseline
            --profile jetson_orin
            --thresholds-file "${CMAKE_CURRENT_BINARY_DIR}/gate_evidence/perf_ctest_hard/thresholds_jetson_orin.json"
            --out-dir "${CMAKE_CURRENT_BINARY_DIR}/gate_evidence/perf_ctest_hard/baseline")
    add_test(
        NAME vectordb_v3_perf_gate_hard_tests
        COMMAND
            ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/perf_gate.py"
            --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
            --mode hard
            --profile jetson_orin
            --thresholds-file "${CMAKE_CURRENT_BINARY_DIR}/gate_evidence/perf_ctest_hard/thresholds_jetson_orin.json"
            --baseline-ref "${CMAKE_CURRENT_BINARY_DIR}/gate_evidence/perf_ctest_hard/baseline/baseline_metrics.json"
            --out-dir "${CMAKE_CURRENT_BINARY_DIR}/gate_evidence/perf_ctest_hard")
    set_tests_properties(vectordb_v3_perf_gate_hard_tests PROPERTIES DEPENDS vectordb_v3_perf_gate_baseline_tests)
    add_test(
        NAME vectordb_v3_reproducibility_tests
        COMMAND
            ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_reproducibility.py"
            --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
            --events-only
            --out-dir "${CMAKE_CURRENT_BINARY_DIR}/gate_evidence/repro_ctest")
endif()
