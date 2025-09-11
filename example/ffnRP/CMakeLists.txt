##
# Transformer Feed-Forward (FFN) Reduced Precision Example
#
# SPDX-License-Identifier: ISC
##

cmake_minimum_required(VERSION 3.25)

set(_TARGET_NAME ffnRP)
project(${_TARGET_NAME} LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)
    if(alpaka_USE_SOURCE_TREE)
        set(alpaka_BUILD_EXAMPLES OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

add_executable(${_TARGET_NAME}
    src/ffnRP.cpp
    src/FfnStage1Kernel.hpp
    src/FfnStage2Kernel.hpp
)
target_include_directories(${_TARGET_NAME} PRIVATE src)
target_link_libraries(${_TARGET_NAME} PUBLIC alpaka::alpaka)
alpaka_finalize(${_TARGET_NAME})
set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER example)

add_test(NAME ${_TARGET_NAME}
    COMMAND ${_TARGET_NAME} --hidden 128 --batch 2 --seq 16 --type float --repeats 1)
