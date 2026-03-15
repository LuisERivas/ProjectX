cmake_minimum_required(VERSION 3.16)
project(vectordb_v3_m1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(vectordb_v3
    src/vector_store.cpp
    src/telemetry.cpp
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

enable_testing()
add_test(NAME vectordb_v3_tests COMMAND vectordb_v3_tests)
