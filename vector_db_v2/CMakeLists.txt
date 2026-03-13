cmake_minimum_required(VERSION 3.16)
project(vectordb_v2_m1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(vectordb_v2
    src/vector_store.cpp
)
target_include_directories(vectordb_v2 PUBLIC include)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_compile_definitions(vectordb_v2 PUBLIC VECTOR_DB_V2_CUDA_ENABLED=1)
else()
    target_compile_definitions(vectordb_v2 PUBLIC VECTOR_DB_V2_CUDA_ENABLED=0)
endif()

add_executable(vectordb_v2_cli
    cli/main.cpp
)
target_link_libraries(vectordb_v2_cli PRIVATE vectordb_v2)

add_executable(vectordb_v2_tests
    tests/test_m1.cpp
)
target_link_libraries(vectordb_v2_tests PRIVATE vectordb_v2)

enable_testing()
add_test(NAME vectordb_v2_tests COMMAND vectordb_v2_tests)
