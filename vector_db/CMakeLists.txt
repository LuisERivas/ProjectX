cmake_minimum_required(VERSION 3.16)
project(vectordb_phase1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(vectordb
    src/vector_store.cpp
)
target_include_directories(vectordb PUBLIC include)

add_executable(vectordb_cli
    cli/main.cpp
)
target_link_libraries(vectordb_cli PRIVATE vectordb)

add_executable(vectordb_tests
    tests/test_phase1.cpp
)
target_link_libraries(vectordb_tests PRIVATE vectordb)

enable_testing()
add_test(NAME vectordb_tests COMMAND vectordb_tests)

