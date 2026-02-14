cmake_minimum_required(VERSION 3.20)
project(md_benchmark)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Benchmark executable (no graphics)
add_executable(benchmark Q1.cpp)

# OpenMP for macOS
target_compile_options(benchmark PRIVATE
    -O3
    -Xpreprocessor -fopenmp
)

target_include_directories(benchmark PRIVATE
    /opt/homebrew/opt/libomp/include
)

target_link_directories(benchmark PRIVATE
    /opt/homebrew/opt/libomp/lib
)

target_link_libraries(benchmark PRIVATE omp)
