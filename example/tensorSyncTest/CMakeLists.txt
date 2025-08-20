cmake_minimum_required(VERSION 3.22)

alpaka_add_executable(
    tensorSyncTest
    src/tensorSyncTest.cpp)

target_link_libraries(
    tensorSyncTest
    PUBLIC alpaka::alpaka)
