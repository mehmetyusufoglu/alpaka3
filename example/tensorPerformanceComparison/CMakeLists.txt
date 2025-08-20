cmake_minimum_required(VERSION 3.22)

alpaka_add_executable(
    tensorPerformanceComparison
    src/tensorPerformanceComparison.cpp)

target_link_libraries(
    tensorPerformanceComparison
    PUBLIC alpaka::alpaka)
