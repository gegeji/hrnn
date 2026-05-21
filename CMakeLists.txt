cmake_minimum_required(VERSION 3.14)
project(hrnn CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)

add_compile_options(-march=native -Wall -Wextra)

find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

include_directories(include)

# HRNN — main method
add_executable(hrnn_build         src/hrnn_build.cpp)
add_executable(hrnn_search        src/hrnn_search.cpp)
add_executable(hrnn_dynamic_test  src/hrnn_dynamic_test.cpp)

# HAMG — graph baseline
add_executable(hamg_build         src/hamg_build.cpp)
add_executable(hamg_search        src/hamg_search.cpp)

# HNSW kNN baseline
add_executable(knn_build          src/knn_build.cpp)
add_executable(knn_search         src/knn_search.cpp)

# Naive RkNN baselines (SFT, RDT)
add_executable(rknn_hnsw_sft_search src/rknn_hnsw_sft_rdt.cpp)

foreach(tgt IN ITEMS
        hrnn_build hrnn_search hrnn_dynamic_test
        hamg_build hamg_search
        knn_build knn_search
        rknn_hnsw_sft_search)
    target_link_libraries(${tgt} Boost::program_options OpenMP::OpenMP_CXX)
endforeach()
