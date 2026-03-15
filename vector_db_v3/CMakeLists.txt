cmake_minimum_required(VERSION 3.16)
project(vectordb_v3_m1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(vectordb_v3
    src/vector_store.cpp
    src/telemetry.cpp
    src/codec/checksum.cpp
    src/codec/io.cpp
    src/codec/artifacts.cpp
)
target_include_directories(vectordb_v3 PUBLIC include)

if (MSVC)
    target_compile_options(vectordb_v3 PRIVATE /W4)
else()
    target_compile_options(vectordb_v3 PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_compile_definitions(vectordb_v3 PUBLIC VECTOR_DB_V3_CUDA_ENABLED=1)
else()
    target_compile_definitions(vectordb_v3 PUBLIC VECTOR_DB_V3_CUDA_ENABLED=0)
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

enable_testing()
add_test(NAME vectordb_v3_tests COMMAND vectordb_v3_tests)
add_test(NAME vectordb_v3_codec_primitives_tests COMMAND vectordb_v3_codec_primitives_tests)
add_test(NAME vectordb_v3_codec_artifacts_tests COMMAND vectordb_v3_codec_artifacts_tests)
add_test(NAME vectordb_v3_codec_corruption_tests COMMAND vectordb_v3_codec_corruption_tests)
add_test(NAME vectordb_v3_durability_wal_tests COMMAND vectordb_v3_durability_wal_tests)
add_test(NAME vectordb_v3_durability_checkpoint_tests COMMAND vectordb_v3_durability_checkpoint_tests)
add_test(NAME vectordb_v3_durability_replay_crash_tests COMMAND vectordb_v3_durability_replay_crash_tests)
add_test(NAME vectordb_v3_durability_corruption_tests COMMAND vectordb_v3_durability_corruption_tests)
